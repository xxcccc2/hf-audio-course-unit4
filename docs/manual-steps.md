# Manual Steps

## Steps that require the user

1. Publish the repository to a Git remote that Colab can access.
2. Open the notebook in Google Colab.
3. Replace the repository URL placeholder in the clone cell.
4. Switch the runtime to GPU.
5. If Hub upload is desired, authenticate with `notebook_login`.
6. Run the AST training cell.
7. Run the optional Hub upload command.

## Steps completed by the agent

- repository files
- training script
- demo script
- notebook runner
- smoke tests
 - local verification
