# Nimbus project notes

This is fictional sample data for trying Atlas document retrieval.

## Architecture

Nimbus is a recipe collection application. The browser frontend uses React and
TypeScript. The backend uses Python. Recipes and meal plans live in PostgreSQL.
The initial application supports recipe links, manual recipe entry, tagging,
and a weekly meal calendar.

## Release plan

The fictional private beta is scheduled for October 16, 2026. The first beta
accepts 25 invited testers. Feedback is collected through a form inside the app.
The beta acceptance criteria include successful recipe creation, tag filtering,
and adding a recipe to a calendar slot.

## Testing

Developers run unit tests for domain rules and integration tests for API endpoints.
Browser tests cover adding recipes and assigning a recipe to a meal-plan slot.
Every pull request must pass the automated test suite before merging.

## Known limitations

Social media links are stored as bookmarks. Automatic video transcription and
recipe extraction are planned future work. The application does not promise that
private or inaccessible social media content can be imported.
