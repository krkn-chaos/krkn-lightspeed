import logging
import os
import json
import tempfile
import subprocess
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaissDocumentIndexer:
    """Documentation indexer using our proven
    FAISS + all-MiniLM-L6-v2 approach"""

    def __init__(self, home_dir: str = "data"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = None
        self.index = None
        self.home_dir = home_dir

    def scrape_krkn_docs(
        self, github_repo: str, repo_path: str
    ) -> List[Dict[str, Any]]:
        """Fetch documentation by cloning GitHub repository"""
        docs = []

        try:
            docs = self._clone_and_extract_docs(github_repo, repo_path)

            # Add chaos testing guide with specific chunking strategy
            chaos_guide = self._fetch_chaos_testing_guide(github_repo)
            if chaos_guide:
                docs.append(chaos_guide)

            logger.info(f"Found {len(docs)} documents from GitHub repository")

        except Exception as e:
            logger.error(f"Error during GitHub repository cloning: {e}")

        return docs

    def _fetch_chaos_testing_guide(self, repo_url: str) -> Dict[str, Any]:
        """Fetch the chaos testing guide specifically
        and set it up for heading-based chunking"""
        guide_path = "content/en/docs/chaos-testing-guide/_index.md"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repository using shared method
                self._clone_repository(repo_url, temp_dir)

                # Read the chaos testing guide file
                guide_file = os.path.join(temp_dir, guide_path)
                if os.path.exists(guide_file):
                    with open(guide_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Parse frontmatter to get title
                    title = "Chaos Testing Guide"
                    if content.startswith("---"):
                        try:
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                frontmatter = parts[1]
                                content_body = parts[2].strip()

                                # Extract title from frontmatter
                                for line in frontmatter.split("\n"):
                                    if line.strip().startswith("title:"):
                                        title = (
                                            line.split(":", 1)[1]
                                            .strip()
                                            .strip("\"'")
                                        )
                                        break

                                content = content_body
                        except Exception:
                            pass

                    return {
                        "url": "https://krkn-chaos.dev/"
                        "docs/chaos-testing-guide/",
                        "title": title,
                        "content": content,
                        "source": guide_path,
                        "github_url": f"https://github.com/"
                        f"krkn-chaos/website/blob/main/{guide_path}",
                        "path": guide_path,
                        "chunking_strategy": "heading",
                        "heading_level": "###",
                    }
                else:
                    logger.warning(
                        f"Chaos testing guide not found at {guide_path}"
                    )

            except Exception as e:
                logger.error(f"Failed to fetch chaos testing guide: {e}")

        return None

    def _clone_repository(self, repo_url: str, temp_dir: str) -> str:
        """Clone repository to directory and return the path"""
        try:
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--quiet",
                    repo_url,
                    temp_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(f"Repository cloned to: {temp_dir}")
            return temp_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository {repo_url}: {e.stderr}")
            raise

    def scrape_krkn_hub_scenarios(
        self, github_repo: str
    ) -> List[Dict[str, Any]]:
        """Fetch krknctl-input.json files from krkn-hub repository"""
        docs = []

        try:
            docs = self._clone_and_extract_scenario_inputs(github_repo)
            logger.info(
                f"Found {len(docs)} scenario input definitions from krkn-hub"
            )

        except Exception as e:
            logger.error(f"Error during krkn-hub repository cloning: {e}")

        return docs

    def _clone_and_extract_docs(
        self, repo_url: str, docs_path: str
    ) -> List[Dict[str, Any]]:
        """Clone repository and extract markdown files from docs directory"""
        docs = []

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repository using shared method
                self._clone_repository(repo_url, temp_dir)

                full_docs_path = os.path.join(temp_dir, docs_path)

                if not os.path.exists(full_docs_path):
                    logger.warning(
                        f"Documentation path not found: {docs_path}"
                    )
                    return docs

                docs = self._extract_markdown_files(full_docs_path, docs_path)

            except Exception as e:
                logger.error(f"Error processing cloned repository: {e}")
                raise

        return docs

    def _clone_and_extract_scenario_inputs(
        self, repo_url: str
    ) -> List[Dict[str, Any]]:
        """Clone krkn-hub repository and extract krknctl-input.json files"""
        docs = []

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repository using shared method
                self._clone_repository(repo_url, temp_dir)

                docs = self._extract_scenario_input_files(temp_dir)

            except Exception as e:
                logger.error(
                    f"Error processing cloned krkn-hub repository: {e}"
                )
                raise

        return docs

    def _extract_scenario_input_files(
        self, base_path: str
    ) -> List[Dict[str, Any]]:
        """Recursively extract krknctl-input.json
        files from krkn-hub directory"""
        docs = []

        try:
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file == "krknctl-input.json":
                        file_path = os.path.join(root, file)
                        # Get scenario name from directory
                        scenario_name = os.path.basename(root)
                        logger.info(
                            f"Processing scenario input file: "
                            f"{scenario_name}/krknctl-input.json"
                        )
                        doc = self._process_scenario_input_file(
                            file_path, scenario_name
                        )
                        if doc:
                            logger.info(
                                f"Successfully processed "
                                f"scenario: {scenario_name}"
                            )
                            docs.append(doc)
                        else:
                            logger.warning(
                                f"Failed to process scenario: {scenario_name}"
                            )

        except Exception as e:
            logger.error(f"Error extracting scenario input files: {e}")

        return docs

    def _process_scenario_input_file(
        self, file_path: str, scenario_name: str
    ) -> Dict[str, Any]:
        """Process individual krknctl-input.json file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                scenario_inputs = json.load(f)

            # Create comprehensive documentation for the scenario
            content_parts = [
                f"# krknctl Scenario: {scenario_name}",
                f"Command: krknctl run {scenario_name}",
                "",
                "## Parameters",
                "",
            ]

            for param in scenario_inputs:
                name = param.get("name", "")
                description = param.get("description", "")
                short_desc = param.get("short_description", "")
                param_type = param.get("type", "")
                default = param.get("default", "")
                required = param.get("required", "false")
                validator = param.get("validator", "")

                # Build parameter documentation
                content_parts.extend(
                    [
                        f"### --{name}",
                        f"**Type:** {param_type}",
                        f"**Required:** {required}",
                        f"**Description:** {description}",
                    ]
                )

                if short_desc and short_desc != description:
                    content_parts.append(
                        f"**Short Description:** {short_desc}"
                    )

                if default:
                    content_parts.append(f"**Default:** {default}")

                if validator:
                    content_parts.append(
                        f"**Validation Pattern:** {validator}"
                    )

                content_parts.extend(["", "---", ""])

            # Add usage example
            content_parts.extend(
                ["## Usage Example", f"krknctl run {scenario_name}", ""]
            )

            # Add parameter examples
            example_params = []
            for param in scenario_inputs:
                name = param.get("name", "")
                default = param.get("default", "")
                if default:
                    example_params.append(f"--{name} {default}")

            if example_params:
                content_parts.append(
                    f"krknctl run {scenario_name} "
                    f"{' '.join(example_params[:3])}"
                )

            content = "\n".join(content_parts)

            return {
                "url": f"https://github.com/krkn-chaos/"
                f"krkn-hub/tree/main/{scenario_name}",
                "title": f"krknctl {scenario_name} - Scenario Parameters",
                "content": content,
                "source": file_path,
                "scenario_name": scenario_name,
                "github_url": f"https://github.com/krkn-chaos/"
                f"krkn-hub/blob/main/{scenario_name}/krknctl-input.json",
            }

        except Exception as e:
            logger.warning(
                f"Failed to process scenario input file {file_path}: {e}"
            )
            return None

    def _extract_markdown_files(
        self, base_path: str, relative_docs_path: str
    ) -> List[Dict[str, Any]]:
        """Recursively extract markdown files from directory"""
        docs = []

        try:
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        # DEBUG: Log all markdown files found
                        logger.info(f"Processing markdown file: {file_path}")
                        doc = self._process_markdown_file(
                            file_path, base_path, relative_docs_path
                        )
                        if doc:
                            # DEBUG: Log successful processing
                            logger.info(
                                f"Successfully processed: "
                                f"{file} -> {doc['title']}"
                            )
                            docs.append(doc)
                        else:
                            # DEBUG: Log failed processing
                            logger.warning(f"Failed to process: {file}")

        except Exception as e:
            logger.error(f"Error extracting markdown files: {e}")

        return docs

    def _process_markdown_file(
        self, file_path: str, base_path: str, relative_docs_path: str
    ) -> Dict[str, Any]:
        """Process individual markdown file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            rel_path = os.path.relpath(file_path, base_path)
            title = (
                os.path.basename(file_path)
                .replace(".md", "")
                .replace("_", " ")
                .title()
            )

            # Parse frontmatter if present
            if content.startswith("---"):
                try:
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1]
                        content_body = parts[2].strip()

                        # Extract title from frontmatter
                        for line in frontmatter.split("\n"):
                            if line.strip().startswith("title:"):
                                title = (
                                    line.split(":", 1)[1].strip().strip("\"'")
                                )
                                break

                        content = content_body
                except Exception:
                    pass

            # Generate URLs compatible with existing structure
            github_path = f"{relative_docs_path}/{rel_path}"
            github_url = (
                f"https://github.com/krkn-chaos"
                f"/website/blob/main/{github_path}"
            )
            docs_url = (
                f"https://krkn-chaos.dev/docs/{rel_path.replace('.md', '/')}"
            )

            return {
                "url": docs_url,
                "title": title,
                "content": content,
                "source": file_path,
                "github_url": github_url,
                "path": github_path,
            }

        except Exception as e:
            logger.warning(f"Failed to process file {file_path}: {e}")
            return None

    def chunk_by_size(
        self, doc: Dict[str, Any], characters: int = 512
    ) -> List[Dict[str, Any]]:
        """Split document by character count"""
        chunks = []
        content = doc["content"]

        # Split into chunks by character count
        for i in range(0, len(content), characters):
            chunk_content = content[i : i + characters]  # NOQA

            if len(chunk_content.strip()) > 50:  # Only keep meaningful chunks
                chunks.append(
                    {
                        "url": doc["url"],
                        "title": doc["title"],
                        "content": chunk_content,
                        "source": doc["source"],
                        "chunk_type": "size_based",
                    }
                )

        return chunks

    def chunk_by_heading(
        self, doc: Dict[str, Any], heading: str = "###"
    ) -> List[Dict[str, Any]]:
        """Split document by heading level (e.g., '###' for H3)"""
        chunks = []
        content = doc["content"]
        lines = content.split("\n")

        current_section = []
        current_heading_title = "Introduction"
        section_number = 0

        for line in lines:
            # Check if this line starts with the specified heading level
            if line.strip().startswith(heading + " "):
                # Save previous section if it has content
                if current_section:
                    section_content = "\n".join(current_section).strip()
                    if (
                        len(section_content) > 100
                    ):  # Only keep substantial sections
                        chunks.append(
                            {
                                "url": f"{doc['url']}#"
                                f"{current_heading_title.lower().replace(' ', '-').replace('/', '-')}",  # NOQA
                                "title": f"{doc['title']}: {current_heading_title}",  # NOQA
                                "content": section_content,
                                "source": doc["source"],
                                "chunk_type": "heading_based",
                                "heading_level": heading,
                                "section_title": current_heading_title,
                                "section_number": section_number,
                            }
                        )

                # Start new section
                current_heading_title = line.strip()[
                    len(heading) :  # NOQA
                ].strip()  # Remove heading markers
                section_number += 1
                current_section = [line]  # Include the heading in the section
            else:
                # Add line to current section
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            section_content = "\n".join(current_section).strip()
            if len(section_content) > 100:
                chunks.append(
                    {
                        "url": f"{doc['url']}#"
                        f"{current_heading_title.lower().replace(' ', '-').replace('/', '-')}",  # NOQA
                        "title": f"{doc['title']}: {current_heading_title}",
                        "content": section_content,
                        "source": doc["source"],
                        "chunk_type": "heading_based",
                        "heading_level": heading,
                        "section_title": current_heading_title,
                        "section_number": section_number,
                    }
                )

        return chunks

    def chunk_documents(
        self, docs: List[Dict[str, Any]], chunk_size: int = 512
    ) -> List[Dict[str, Any]]:
        """Split documents into smaller chunks
        using their specified chunking strategy"""
        chunked_docs = []

        for doc in docs:
            strategy = doc.get("chunking_strategy", "size")

            if strategy == "heading":
                heading_level = doc.get("heading_level", "###")
                chunks = self.chunk_by_heading(doc, heading=heading_level)
                logger.info(
                    f"Applied heading-based chunking "
                    f"({heading_level}) to {doc['title']}: "
                    f"{len(chunks)} chunks"
                )
            else:
                # Default to size-based chunking
                chunks = self.chunk_by_size(doc, characters=chunk_size)

            chunked_docs.extend(chunks)

            # DEBUG: Log krknctl specific documents
            if (
                "krknctl" in doc["title"].lower()
                or "krknctl" in doc.get("source", "").lower()
            ):
                logger.info(
                    f"KRKNCTL DOC CHUNKED: {doc['title']} "
                    f"(strategy: {strategy}, chunks: {len(chunks)})"
                )

        # Add chunk_id to all chunks
        for i, chunk in enumerate(chunked_docs):
            chunk["chunk_id"] = i

        return chunked_docs

    def create_embeddings(self, docs: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for document chunks"""
        logger.info(f"Creating embeddings for {len(docs)} document chunks")

        texts = [doc["content"] for doc in docs]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(
            dimension
        )  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        return index

    def save_index(
        self,
        docs: List[Dict[str, Any]],
        embeddings: np.ndarray,
        index: faiss.Index,
        output_dir: str,
    ):
        """Save the complete index to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save FAISS index
        faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

        # Save documents metadata
        with open(os.path.join(output_dir, "documents.json"), "w") as f:
            json.dump(docs, f, indent=2)

        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

        logger.info(f"Index saved to {output_dir}")

    def build_and_save_index(
        self,
        github_repo: str,
        repo_path: str,
        output_dir: str,
        krkn_hub_repo: str = None,
    ):
        """Build complete index from GitHub repo
        and optionally krkn-hub scenarios"""
        logger.info(f"Building index from {github_repo}/{repo_path}")

        # Scrape main documentation
        docs = self.scrape_krkn_docs(github_repo, repo_path)
        if not docs:
            raise Exception("No documents found to index")

        # Scrape krkn-hub scenarios if provided
        if krkn_hub_repo:
            logger.info(f"Also indexing scenarios from {krkn_hub_repo}")
            scenario_docs = self.scrape_krkn_hub_scenarios(krkn_hub_repo)
            logger.info(f"Found {len(scenario_docs)} scenario definitions")
            docs.extend(scenario_docs)

        logger.info(f"Total documents to index: {len(docs)}")

        # Chunk documents
        chunked_docs = self.chunk_documents(docs)
        logger.info(f"Created {len(chunked_docs)} document chunks")

        # Create embeddings
        embeddings = self.create_embeddings(chunked_docs)

        # Build FAISS index
        index = self.build_faiss_index(embeddings)

        # Save everything
        self.save_index(chunked_docs, embeddings, index, output_dir)

        return chunked_docs, embeddings, index
