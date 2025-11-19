#!/usr/bin/env python3
"""
Build FAISS index at container build time
"""
import os
import sys

# Add /app to path if running in container
sys.path.insert(0, '/app')

from utils.faiss_document_indexer import FaissDocumentIndexer  # noqa: E402


def main():
    """Build FAISS index from documentation sources"""
    print("="*60)
    print("Building FAISS index at build time...")
    print("="*60)

    # Configuration from environment or defaults
    github_repo = os.getenv('GITHUB_REPO',
                            'https://github.com/krkn-chaos/website')
    repo_path = os.getenv('REPO_PATH', 'content/en/docs')
    output_dir = os.getenv('PERSIST_DIR', '/app/faiss_index')
    krkn_hub_repo = os.getenv('KRKN_HUB_REPO',
                              'https://github.com/krkn-chaos/krkn-hub')
    krkn_hub_branch = os.getenv('KRKN_HUB_BRANCH', None)

    print(f"GitHub Repo: {github_repo}")
    print(f"Repo Path: {repo_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Krkn Hub Repo: {krkn_hub_repo}")
    print("="*60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build index
    indexer = FaissDocumentIndexer()
    indexer.build_and_save_index(
        github_repo=github_repo,
        repo_path=repo_path,
        output_dir=output_dir,
        krkn_hub_repo=krkn_hub_repo,
        krkn_hub_branch=krkn_hub_branch
    )

    print("="*60)
    print("FAISS index built successfully!")
    print(f"Index location: {output_dir}")

    # Verify index files exist
    index_file = os.path.join(output_dir, 'index.faiss')
    if os.path.exists(index_file):
        size_mb = os.path.getsize(index_file) / (1024 * 1024)
        print(f"Index file size: {size_mb:.2f} MB")
    else:
        print("WARNING: Index file not found!")
        sys.exit(1)

    print("="*60)


if __name__ == '__main__':
    main()
