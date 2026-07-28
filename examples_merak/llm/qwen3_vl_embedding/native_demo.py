import argparse

from native_utils import embed_native_items, load_native_embedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the native Qwen3-VL-Embedding model."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "A dog playing in the park",
            "A cat sitting on a chair",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    embedder = load_native_embedder(args.model_dir)
    embeddings = embed_native_items(
        embedder,
        [{"text": text} for text in args.text],
    )
    similarities = embeddings @ embeddings.T
    print(f"embedding_shape: {tuple(embeddings.shape)}")
    print(f"embedding_norms: {embeddings.norm(dim=-1).tolist()}")
    print("similarity:")
    print(similarities)


if __name__ == "__main__":
    main()
