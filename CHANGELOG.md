# Release Notes

## 0.8.1 - (released on 24-02-13)

Bugfix release with changes aimed to increase stability.

- Further improvements to document processing, including an update in document demo functionality and the introduction of functional programming approaches to document generation.
- Enhanced type hinting across the system, ensuring better code quality and readability.
- Updated several type definitions for increased accuracy and clarity.
- Refined error handling mechanisms for more robust operation under various conditions.
- Advanced our documentation generation, making it more comprehensive and easier to navigate.
- Made adjustments to accommodate Pydantic v2 changes, ensuring compatibility and leveraging new features.
- Enhanced pipeline graphs for improved visualization and analysis.

### Bug Fixes

- Addressed multiple small bugs and typing errors identified in the codebase, enhancing system stability.
- Fixed specific issues with multi-output handling in the code, ensuring correct operation in diverse scenarios.
- Resolved problems with typed class recognition, improving the system's ability to handle complex data structures.
- Repaired and updated documentation, making it more accurate and useful for users.
- Eliminated a return overload issue in FunctionOperator, simplifying the code structure.
- Fixed a minor bug in type declarations, ensuring consistency and correctness.
- Made clone_data executable, enhancing its usability in practical scenarios.

### Miscellaneous

- Refactored operator type hints and other parts of the code for cleaner, more efficient operation.
- Removed more deprecated documentation related to agents, focusing on the streamlined functionality of the current system.
- Improved offline handling capabilities, offering better performance in environments without constant internet access.

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
