"""
Text Converter

Converts structured deed data to natural language text documents
for use with text-based RAG systems (Vector RAG, LightRAG).

Supports multiple text styles:
- concise: Compact, structured format
- narrative: Natural prose style
- mixed: Combination of both
"""

import random
from typing import Dict, List, Any, Optional
from pathlib import Path


class TextConverter:
    """
    Converts structured deed data to text documents.

    Each deed is converted to a text representation that
    preserves all queryable information.
    """

    def __init__(self, style: str = "mixed", style_ratio: float = 0.6, seed: int = 42):
        """
        Initialize converter.

        Args:
            style: "concise", "narrative", or "mixed"
            style_ratio: For mixed, ratio of concise to narrative (0.6 = 60% concise)
            seed: Random seed for mixed style selection
        """
        self.style = style
        self.style_ratio = style_ratio
        self.rng = random.Random(seed)

    def convert_deed(self, deed: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Convert a single deed to text.

        Args:
            deed: Deed dictionary with all fields
            style: Override style for this deed

        Returns:
            Text representation of the deed
        """
        use_style = style or self._pick_style()

        if use_style == "concise":
            return self._to_concise(deed)
        elif use_style == "narrative":
            return self._to_narrative(deed)
        else:
            return self._to_concise(deed)

    def _pick_style(self) -> str:
        """Pick style for mixed mode."""
        if self.style == "mixed":
            return "concise" if self.rng.random() < self.style_ratio else "narrative"
        return self.style

    def _to_concise(self, deed: Dict[str, Any]) -> str:
        """Convert deed to concise format."""
        lines = [
            f"DEED: {deed['deed_id']}",
            f"Signed: {deed['signed_date']} | Recorded: {deed['recorded_date']}",
            f"Location: {deed['street_name']}, {deed['subdivision_name']}",
            f"From: {deed['grantor_name']} To: {deed['grantee_name']}",
            f"Plan Book: {deed['plan_book']}, Page: {deed['plan_page']}"
        ]

        if deed.get('has_covenant'):
            lines.append(f"COVENANT: {deed.get('covenant_text', 'Present')}")

        lines.append(f"Review Status: {deed.get('review_status', 'pending')}")

        return "\n".join(lines)

    def _to_narrative(self, deed: Dict[str, Any]) -> str:
        """Convert deed to narrative format."""
        year = deed['signed_year']

        parts = [
            f"Deed {deed['deed_id']}: On {self._format_date(deed['signed_date'])}, "
            f"{deed['grantor_name']} conveyed property located at {deed['street_name']} "
            f"in the {deed['subdivision_name']} subdivision to {deed['grantee_name']}. "
            f"The deed was officially recorded on {self._format_date(deed['recorded_date'])} "
            f"in Plan Book {deed['plan_book']}, Page {deed['plan_page']}."
        ]

        if deed.get('has_covenant'):
            covenant_text = deed.get('covenant_text', 'a restrictive covenant')
            parts.append(f" This deed contains a racial restrictive covenant stating: \"{covenant_text}\"")

        parts.append(f" Review status: {deed.get('review_status', 'pending')}.")

        return "".join(parts)

    def _format_date(self, date_str: str) -> str:
        """Format date string as readable text."""
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_str)
            return dt.strftime("%B %d, %Y")
        except Exception:
            return date_str

    def convert_all(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert all deeds to text.

        Args:
            data: Full dataset from SyntheticDeedGenerator

        Returns:
            Dict mapping deed_id to text
        """
        texts = {}
        for deed_id, deed in data.get('deeds', {}).items():
            texts[deed_id] = self.convert_deed(deed)
        return texts

    def save_as_documents(self, data: Dict[str, Any], output_dir: str) -> List[str]:
        """
        Save each deed as a separate text file.

        Args:
            data: Full dataset
            output_dir: Directory for output files

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = []
        texts = self.convert_all(data)

        for deed_id, text in texts.items():
            file_path = output_path / f"{deed_id}.txt"
            with open(file_path, 'w') as f:
                f.write(text)
            files.append(str(file_path))

        return files

    def save_as_single_file(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Save all deeds as a single combined text file.

        Args:
            data: Full dataset
            output_path: Path for output file

        Returns:
            Path to created file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        texts = self.convert_all(data)

        # Combine with clear separators
        combined = "\n\n" + "=" * 60 + "\n\n"
        combined = combined.join(texts.values())

        with open(path, 'w') as f:
            f.write(combined)

        return str(path)

    def to_document_list(self, data: Dict[str, Any]) -> List[str]:
        """
        Convert to list of text documents.

        Args:
            data: Full dataset

        Returns:
            List of text documents (one per deed)
        """
        texts = self.convert_all(data)
        return list(texts.values())


if __name__ == "__main__":
    # Quick test
    from synthetic_generator import SyntheticDeedGenerator, GeneratorConfig

    config = GeneratorConfig(num_deeds=5, seed=42)
    gen = SyntheticDeedGenerator(config)
    data = gen.generate()

    print("=== CONCISE STYLE ===")
    converter = TextConverter(style="concise")
    sample_deed = list(data['deeds'].values())[0]
    print(converter.convert_deed(sample_deed))

    print("\n=== NARRATIVE STYLE ===")
    converter = TextConverter(style="narrative")
    print(converter.convert_deed(sample_deed))

    print("\n=== MIXED STYLE (5 deeds) ===")
    converter = TextConverter(style="mixed")
    for i, (deed_id, text) in enumerate(converter.convert_all(data).items()):
        print(f"\n--- {deed_id} ---")
        print(text[:200] + "..." if len(text) > 200 else text)
