import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
import torch

print('Loading GR00T N1.6-DROID...')
with torch.no_grad():
    policy = Gr00tPolicy(
        model_path='/data02/datasets/GR00T-N1.6-DROID',
        embodiment_tag=EmbodimentTag.OXE_DROID,
        device='cuda',
    )

    obs = {
        'video': {
            'exterior_image_1_left': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
            'exterior_image_2_left': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
            'wrist_image_left': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
        },
        'state': {
            'joint_position': np.random.rand(1, 1, 7).astype(np.float32),
            'gripper_position': np.random.rand(1, 1, 1).astype(np.float32),
        },
        'language': {
            'annotation.language.language_instruction': [['pick up the red apple']],
        },
    }

    action = policy.get_action(obs)
    print('Action output:')
    for k, v in action[0].items():
        print(f'  {k}: shape={v.shape}')
    print('GR00T Inference Success!')