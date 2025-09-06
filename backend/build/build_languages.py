from tree_sitter import Language

Language.build_library(
    # Output path for the DLL
    "languages.dll",
    [
        "tree-sitter-javascript",
        "tree-sitter-typescript/typescript",
        "tree-sitter-typescript/tsx",
        "tree-sitter-java",
    ]
)
