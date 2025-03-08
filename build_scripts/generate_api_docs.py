"""Generate the code reference pages."""

import logging
from pathlib import Path

import mkdocs_gen_files

logger = logging.getLogger(__name__)

nav = mkdocs_gen_files.Nav()
root = Path("src")  # / Path("pydvl")
for path in sorted(root.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    parts = tuple(module_path.parts)

    extra_preamble = None
    if parts[:2] == ("pydvl", "value"):
        extra_preamble = (
            '!!! Danger "Deprecation notice"\n'
            "    This module is deprecated since v0.10.0"
            "    in favor of [pydvl.valuation][].\n"
        )
        full_doc_path = Path("deprecated") / doc_path
    elif parts[:2] == ("pydvl", "parallel"):
        extra_preamble = (
            '!!! Danger "Deprecation notice"\n'
            "    This module is deprecated since v0.10.0 in favor of"
            "    joblib's context manager [joblib.parallel_config][].\n"
        )

        full_doc_path = Path("deprecated") / doc_path
    else:
        full_doc_path = Path("api") / doc_path

    extra_args = ""
    if parts[-1] == "__init__":
        logger.error(
            f"Generating API docs for {module_path} with last part {parts[-1]}"
        )
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        extra_args = '    selection:\n      members: false\n      filters: ["NONE"]\n'
    elif parts[-1] == "__main__":
        continue
    elif parts[-1].startswith("_"):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        if extra_preamble:
            fd.write(extra_preamble)
        fd.write(f"::: {identifier}")
        if extra_args:
            fd.write("\n")
            fd.write(extra_args)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
