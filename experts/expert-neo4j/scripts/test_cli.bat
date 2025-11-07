@echo off
echo ==============================================================================
echo EXPERT-NEO4J CLI INFERENCE TEST
echo ==============================================================================
echo.

cd /d F:\Node\hivellm\expert

echo [1/3] Testing: Simple relationship query
echo Question: Find all actors and movies
echo.
cli\target\release\expert-cli.exe chat --experts neo4j --max-tokens 100 --device cuda --prompt "/expert neo4j" --prompt "Find all actors and the movies they acted in. Schema: Node properties: Person (name: STRING), Movie (title: STRING). Relationships: (:Person)-[:ACTED_IN]->(Movie)"
echo.

echo ==============================================================================
echo [2/3] Testing: COUNT aggregation
echo Question: Count products per user
echo.
cli\target\release\expert-cli.exe chat --experts neo4j --max-tokens 100 --device cuda --prompt "Count how many products each user purchased. Schema: User (name), Product (name), Relationship: PURCHASED."
echo.

echo ==============================================================================
echo [3/3] Testing: Basic MATCH
echo Question: Find people older than 30
echo.
cli\target\release\expert-cli.exe chat --experts neo4j --max-tokens 100 --device cuda --prompt "Find all people older than 30. Schema: Person (name, age)"
echo.

echo ==============================================================================
echo Test complete!
pause

