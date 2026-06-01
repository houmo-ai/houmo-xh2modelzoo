# Copyright 2025 HOUMO AI
#
# File: graph_module_pipe.py
# Description:
#   Graph module pipeline utilities for xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import defaultdict
from types import MethodType
from typing import Dict, List, Set

import torch
import torch.nn as nn
from torch import fx
from torch.distributed.pipelining import Pipe
from torch.distributed.pipelining._unflatten import _outline_submodules
from torch.fx import GraphModule
from torch.fx.passes.split_module import split_module
from xhquant.frontend.torchfx import _pipe_split_fx

from .logger import get_root_logger

aten_pipe_split_alias = torch.ops.pippy._pipe_split.default
from torch.export.unflatten import (
    InterpreterModule,
    _assign_attr,
    _AttrKind,
    _sink_params,
)


def split_graph_module(graph_module: GraphModule):
    logger = get_root_logger()

    part_idx = 0

    def split_callback(n: fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == (
            "call_function",
            aten_pipe_split_alias,
        ):
            logger.debug(f"Found pipe_split {part_idx}")  # noqa: G004
            part_idx += 1
        return part_idx

    # TODO: what does split do with module invocations? does it move the modules
    # into the submodules?
    split: GraphModule = split_module(graph_module, None, split_callback)
    # a (custom) tracer can produce dead code like orphan get_attr nodes
    split.graph.eliminate_dead_code()
    split.__class__ = graph_module.__class__
    split.recompile()
    return split


def _create_pipe(split: GraphModule):
    logger = get_root_logger()
    # peephole to remove pipe_split
    for submodule in split.modules():
        if isinstance(submodule, fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == (
                    "call_function",
                    aten_pipe_split_alias,
                ):
                    submodule.graph.erase_node(node)
            submodule.recompile()

    # for name, submodule in split.named_children():
    #     if isinstance(submodule, fx.GraphModule):
    #         new_submod = _outline_submodules(submodule.graph)
    #         # Replace old submod
    #         split.register_module(name, new_submod)

    # TODO: backport this into split_module
    def delete_user_reference(node, user):
        """
        Delete reference of `node` from `user`'s arg list.
        Args:
            - node: a `get_attr` node at root.
            - user: a submodule node that uses `node`.
        """
        assert len(user.kwargs) == 0
        use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
        assert len(use_idxs) == 1
        args_copy = list(user.args)
        args_copy.pop(use_idxs[0])
        user.args = tuple(args_copy)
        logger.debug(f"Deleted {node} from user {user}, arg index = {use_idxs[0]}")  # noqa: G004

    # A list of param referrals for deferred deletion.
    # To be accumulated in `move_param_to_callee`.
    to_delete = list()

    def _recursive_getattr_with_parent(mod, fqn):
        # Returns getattr call given a nested FQN, and the last parent
        atoms = fqn.split(".")
        for atom in atoms[:-1]:
            if not hasattr(mod, atom):
                return None, None
            mod = getattr(mod, atom)
        if not hasattr(mod, atoms[-1]):
            return mod, None
        attr = getattr(mod, atoms[-1])
        return mod, attr

    def move_param_to_callee(
        root,
        callee_name,
        param_fqn,
    ):
        """
        Move a parameter from the root module to a submodule.
        Args:
            root: The root module.
            callee_name: The name of the submodule to move the parameter to.
            param_fqn: The fully qualified name of the parameter to move.
        """
        # `atoms` is a list of strings representing the path to the
        # parameter in the original model
        atoms = param_fqn.split(".")
        mod_itr, param_val = _recursive_getattr_with_parent(split, param_fqn)
        # Check whether the parameter is a buffer or a parameter
        is_buffer = atoms[-1] in mod_itr._buffers

        # Check whether the parameter is a tensor
        assert isinstance(
            param_val, torch.Tensor
        ), f"Expected '{param_fqn}' to be {torch.Tensor} but got {type(param_val)}." + (
            f" It might happen if module '{param_fqn}' was passed to some 'leaf function'"
            f"(see https://pytorch.org/docs/stable/fx.html#fx.wrap). Please inspect "
            f"usages of '{param_fqn}' in the traced graph."
            if isinstance(param_val, torch.nn.Module)
            else ""
        )

        # Get submodule
        callee = root.get_submodule(callee_name)
        assert not hasattr(callee, param_fqn), f"Module {callee_name} already has a parameter named {param_fqn}"

        # Assign the parameter to the submodule
        if is_buffer:
            _assign_attr(
                param_val,
                callee,
                param_fqn,
                attr_kind=_AttrKind.BUFFER,
                persistent=True,  # TODO: handle non-persistent buffer
            )
        else:
            _assign_attr(
                param_val,
                callee,
                param_fqn,
                attr_kind=_AttrKind.PARAMETER,
            )
        logger.debug(f"Moved parameter {param_fqn} to {callee_name}")  # noqa: G004

        # Next step is to replace placeholder of submodule with a get_attr.
        # Those placeholders are created by `split_module` inside each
        # submodule.
        # Update: this step is now moved to `_sink_params` because
        # `_sink_params` can do it recursively (i.e. for modules inside
        # submodule)

        to_delete.append((mod_itr, atoms[-1]))

    mod = split
    # [aliasing] store tensor id -> list of FQNs, built from state dict
    # Also assign non-persistent buffers
    id_to_fqns: Dict[int, Set[str]] = defaultdict(set)
    for fqn, tensor in mod.state_dict(keep_vars=True).items():
        id_to_fqns[id(tensor)].add(fqn)
    for fqn, tensor in mod.named_buffers():
        id_to_fqns[id(tensor)].add(fqn)

    # Get the list of all parameters in the root module
    attr_nodes = list(filter(lambda n: n.op == "get_attr", split.graph.nodes))
    for node in attr_nodes:
        # Check whether the parameter is used in only one submodule
        if len(node.users) > 1:
            logger.info(f"Parameter {node.target} used in multiple stages: {node.users}.")  # noqa: G004
        for user in node.users:
            assert user.op == "call_module"
            # Move parameter into submodule
            move_param_to_callee(
                split,
                user.target,
                node.target,
            )

    # After moving the params to their corresponding hierarchies, we also
    # need to move the `get_attr` nodes from the root of the graph to those
    # hierarchies.
    # [aliasing] use id -> fqn mapping to list out all valid FQNs
    inputs_to_state: Dict[str, List[str]] = {}
    for attr in attr_nodes:
        _, tensor = _recursive_getattr_with_parent(mod, attr.target)
        fqns = list(id_to_fqns[id(tensor)])
        if fqns:
            inputs_to_state[attr.name] = fqns
        else:
            assert False, f"Cannot find FQN for {attr.target}"
        # elif attr.target in exported_program.constants:  # lifted constants
        #     inputs_to_state[attr.name] = [attr.target]

    # [aliasing] for each submodule split, assign attributes on FQNs that may be used.
    # We determine this based on whether or not the FQN attribute parent exists.
    # i.e. if the last submodule exists, assign the attribute.
    added_attributes: Dict[str, List[str]] = defaultdict(list)
    for fqn, tensor in mod.state_dict(keep_vars=True).items():
        for name, submod in split.named_children():
            if isinstance(submod, fx.GraphModule):
                parent, child = _recursive_getattr_with_parent(submod, fqn)
                if parent and child is None:  # parent exists, attribute doesn't -> assign
                    added_attributes[name].append(fqn)
                    setattr(parent, fqn.split(".")[-1], tensor)

    # Deferral deletion: Remove the original attributes (to params) from the
    # root GraphModule
    for mod_itr, last_atom in to_delete:
        try:
            delattr(mod_itr, last_atom)
        except AttributeError:
            # This is expected if the parameter is used in multiple stages
            pass

    # This is done by (1) `_sink_params` at each submodule;
    for name, submod in split.named_children():
        if isinstance(submod, fx.GraphModule):
            _sink_params(submod, inputs_to_state, [])
            submod.graph.lint()
            submod.recompile()

    # [aliasing] This step is not super necessary, but helps reduce parameter usage/memory.
    # After _sink_params() routine has run, clean up unused attributes that we previously added.
    # Determine this based on the get_attr nodes - if not used, remove it.
    for name, attributes in added_attributes.items():
        submod = getattr(split, name)
        unused_attributes = set(attributes)
        # track used attributes in the submodule, running DFS on subgraph hierarchy
        stack = [("", submod)]  # (scope, submodule)
        while stack:
            scope, _mod = stack.pop()
            if isinstance(_mod, (fx.GraphModule, InterpreterModule)):
                for node in _mod.graph.nodes:
                    if node.op == "get_attr":
                        # get_attr might get access deeper level attribute
                        fqn = scope + "." + node.target if scope else node.target
                        if fqn in unused_attributes:  # used, remove it
                            unused_attributes.remove(fqn)
            for _name, _submod in _mod.named_children():
                stack.append((scope + "." + _name if scope else _name, _submod))
        # delete unused attributes
        for attr in unused_attributes:
            mod_itr, atoms = submod, attr.split(".")
            for atom in atoms[:-1]:
                mod_itr = getattr(mod_itr, atom)
            delattr(mod_itr, atoms[-1])

    for node in attr_nodes:
        # And (2): remove `get_attr` node from submod's arg list
        for user in copy.copy(node.users):
            assert user.op == "call_module"
            delete_user_reference(node, user)
        # And (3): remove the `get_attr` node from the root graph.
        split.graph.erase_node(node)

    split.delete_all_unused_submodules()
    split.graph.lint()
    split.recompile()

    num_stages = Pipe._number_and_count_forward_stages(split)

    has_loss_and_backward = False
    generated_loss_spec = None

    # has_loss_and_backward = False
    # generated_loss_spec = output_loss_value_spec

    # if output_loss_value_spec is not None:
    #     loss_node, output_node, generated_loss_spec = _find_loss_output(mod, split.graph, output_loss_value_spec)
    #     if loss_node is not None:
    #         _insert_stage_symbolic_backward(
    #             split.graph,
    #             loss_node,
    #             output_node,
    #         )
    #         split.recompile()
    #         has_loss_and_backward = True
    #         logger.debug("Pipeline is in training mode, backward pass generated")
    #     else:
    #         raise RuntimeError(f"Did not find any loss value according to {output_loss_value_spec=}")
    # else:
    #     logger.debug("Pipeline is in inference mode, backward pass not generated")

    logger.debug("Full pipe model:\n" f"{split}")  # noqa: G004

    return Pipe(
        split,
        num_stages,
        has_loss_and_backward,
        generated_loss_spec,
    )


def _split_after_forward(self, *args, **kwargs):
    try:
        return self._orig_forward(*args, **kwargs)
    finally:
        _pipe_split_fx(args[0])


def annotate_pipe_split(model: nn.Module, split_modules: List[nn.Module]):
    for _, mod_to_wrap in model.named_modules():
        if isinstance(mod_to_wrap, tuple(split_modules)):
            mod_to_wrap._orig_forward = mod_to_wrap.forward
            mod_to_wrap._orig_forward = mod_to_wrap.forward
            mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)


def create_pipe(graph_module: GraphModule, num_stages: int):
    pip_spit_ops_cnt = 0
    for node in graph_module.graph.nodes:
        if (node.op, node.target) == (
            "call_function",
            aten_pipe_split_alias,
        ):
            pip_spit_ops_cnt += 1
    assert pip_spit_ops_cnt >= num_stages - 1, f"pipe_split count is {pip_spit_ops_cnt}, but num_stages is {num_stages}"

    step = (pip_spit_ops_cnt + num_stages - 1) // num_stages

    split_node_idx = 0
    part_idx = 0
    del_nodes = []
    for node in graph_module.graph.nodes:
        if (node.op, node.target) == (
            "call_function",
            aten_pipe_split_alias,
        ):
            split_node_idx += 1
            if split_node_idx % step != 0:
                del_nodes.append(node)
            else:
                part_idx < (num_stages - 1)
                part_idx += 1

            if part_idx > (num_stages - 1):
                del_nodes.append(node)

    for node in del_nodes:
        graph_module.graph.erase_node(node)
    graph_module.recompile()

    pip_spit_ops_cnt = 0
    for node in graph_module.graph.nodes:
        if (node.op, node.target) == (
            "call_function",
            aten_pipe_split_alias,
        ):
            pip_spit_ops_cnt += 1

    assert (
        pip_spit_ops_cnt == num_stages - 1
    ), f"The expected number of pipe_split is {pip_spit_ops_cnt}, but get {num_stages}"
    split = split_graph_module(graph_module)
    return _create_pipe(split)
