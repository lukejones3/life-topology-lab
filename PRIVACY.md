# Privacy model

## Never committed

- raw conversations, messages, emails, browser history, media history, health
  data, financial data, contacts, or location history;
- real names, handles, email addresses, phone numbers, account identifiers, or
  relationship labels;
- daily embeddings or reconstructed text;
- event dates and personal timelines;
- model outputs describing the subject;
- credentials, tokens, cookies, exports, databases, or vector stores.

## Repository contents

- algorithms;
- mathematical definitions;
- generic schemas;
- operation-label rubrics;
- test and validation methodology;
- synthetic examples.

## Output discipline

Generated results default to an ignored `output/` directory. Data paths must
point outside the repository. Before every push, run:

```bash
./scripts/privacy_scan.sh
```

The scan checks tracked and untracked repository content for common identifiers,
absolute home-directory paths, credentials, and accidentally copied data files.

