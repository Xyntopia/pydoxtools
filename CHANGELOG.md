# Release Notes

## 0.8.0 - (released on 24-01-28)

some significant updates. main goals were:

- remove support for agents, the functionality was moved to another app:

    (taskyon.space)[https://taskyon.space]

### Enhancements

- Improved page template generation for more efficient document processing.
- Enhanced document graph generation for better visualization and analysis.
- Improved pipeline management for more efficient data processing.
- Added support for more graph objects and improved labeling of graph nodes.

### Bug Fixes

- Made pydoxtools compatible with Spacy > 3.7.
- Added pydantic-settings due to pydantic update.
- Resolved several bugs in page template generation.
- Fixed issues with pandoc tests.
- Addressed multiple small bugs found in various modules.
- Repaired context function for improved stability.
- Fixed issues related to tables functionality.

### Miscellaneous

- Removed deprecated agent and submodule components.
- Replaced testdata submodule with a download script for streamlined setup.
- Updated visualization capabilities for non-pdf pages.