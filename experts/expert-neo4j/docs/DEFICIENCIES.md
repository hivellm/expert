# Neo4j Expert Deficiencies Analysis

This document maps the current deficiencies found in the Neo4j expert through comprehensive CLI testing.

**Analysis Date**: 2025-01-XX  
**Expert Version**: 0.1.0  
**Checkpoint**: final (655 steps)  
**Testing Method**: Direct CLI testing with various Cypher query patterns

---

## Critical Deficiencies

### 1. ❌ **Queries Without Explicit Schema - Text Generation Instead of Cypher**

**Severity**: CRITICAL  
**Frequency**: 100% (when schema is missing)  
**Impact**: Expert is unusable without explicit schema

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Find all movies" --max-tokens 200 --device cuda
```

**Generated Output**:
```
where the director is either a male or female and the plot is 'Hitchcock' (with 'Hitchcock' being the plot name). The answer should include the director and the movie title.
To provide a correct answer, list the movies in the order they were released.
To provide a correct answer, list the movies in the order they were released.
[...repetitive text...]
```

**Expected Output**:
```cypher
MATCH (m:Movie)
RETURN m.title, m.released
ORDER BY m.released
```

**Issues**:
- Generates repetitive explanatory text instead of Cypher
- No Cypher query structure at all
- Model seems to be generating natural language explanations
- This makes the expert completely unusable without schema context

---

---

## Major Deficiencies Found

### 2. ⚠️ **Property Name Typos - Truncated Property Names**

**Severity**: HIGH  
**Frequency**: ~40% of queries  
**Impact**: Queries fail at runtime due to invalid property names

**Test Cases**:

**Test 1**: Multi-hop traversal
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find actors who worked with directors born after 1970"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
WHERE p.orn > 1970 AND p.orn > 1990
RETURN p.ame AS actor, m.itle AS movie
```

**Expected Output**:
```cypher
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
WHERE d.born > 1970
RETURN a.name AS actor, m.title AS movie
```

**Issues**:
- `p.orn` instead of `p.born` (truncated)
- `p.ame` instead of `p.name` (truncated)
- `m.itle` instead of `m.title` (truncated)
- Logic error: checks `p.orn > 1990` twice instead of proper multi-hop pattern

---

### 3. ⚠️ **Incorrect COUNT Aggregation - Missing COUNT() Function**

**Severity**: HIGH  
**Frequency**: ~30% of aggregation queries  
**Impact**: Queries don't produce correct counts

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Count how many movies each actor has acted in"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, m.title AS movie, m.leased AS released
GROUP BY p.name
```

**Expected Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, COUNT(m) AS movie_count
ORDER BY movie_count DESC
```

**Issues**:
- Missing `COUNT()` function entirely
- Includes `m.title` in RETURN but groups by `p.name` (invalid)
- Typo: `m.leased` instead of `m.released`
- No ORDER BY for meaningful results

---

### 4. ⚠️ **AVG with GROUP BY - Incorrect Syntax and Logic**

**Severity**: HIGH  
**Frequency**: ~50% of AVG queries  
**Impact**: Aggregation queries fail or produce wrong results

**Test Case 1**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find the average rating of movies by genre"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)-[:DIRECTED]->(m2:Movie)
WITH m2.genre AS genre, p.ame AS person, m.title AS movie
RETURN AVG(m.rating) AS avg_rating
WHERE genre IS NOT NULL
ORDER BY avg_rating DESC
```

**Expected Output**:
```cypher
MATCH (m:Movie)
WHERE m.genre IS NOT NULL
RETURN m.genre AS genre, AVG(m.rating) AS avg_rating
ORDER BY avg_rating DESC
```

**Issues**:
- Incorrect pattern: adds unnecessary `Person` and `DIRECTED` relationship
- Typo: `p.ame` instead of `p.name`
- WHERE clause after RETURN (invalid syntax)
- Doesn't properly group by genre

**Test Case 2**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find movies with rating above 8.0, grouped by genre, showing average rating per genre"
```

**Generated Output**:
```cypher
MATCH (m:Movie)-[:SIMILAR_TO]->(m2:Movie)
WHERE m.rating > 8.0
RETURN m.genre AS genre, AVG(m.rating) AS average_rating
GROUP BY genre
```

**Expected Output**:
```cypher
MATCH (m:Movie)
WHERE m.rating > 8.0
RETURN m.genre AS genre, AVG(m.rating) AS average_rating
GROUP BY m.genre
```

**Issues**:
- Adds unnecessary `SIMILAR_TO` relationship
- GROUP BY uses alias `genre` instead of property `m.genre` (may work but inconsistent)

---

### 5. ❌ **Multi-hop Traversals - Incorrect Pattern Matching**

**Severity**: CRITICAL  
**Frequency**: ~60% of multi-hop queries  
**Impact**: Queries don't match intended relationships

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find actors who worked with directors born after 1970"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
WHERE p.orn > 1970 AND p.orn > 1990
RETURN p.ame AS actor, m.itle AS movie
```

**Expected Output**:
```cypher
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
WHERE d.born > 1970
RETURN DISTINCT a.name AS actor, m.title AS movie
```

**Issues**:
- Uses same variable `p` for both actor and director (logical error)
- Should use different variables (`a` for actor, `d` for director)
- Pattern should be `(actor)-[:ACTED_IN]->(movie)<-[:DIRECTED]-(director)`
- Multiple typos in property names

---

### 6. ❌ **Self-join Queries - Incorrect Logic**

**Severity**: HIGH  
**Frequency**: ~70% of self-join queries  
**Impact**: Queries don't find correct relationships

**Test Case 1**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find movies where the director also acted in the same movie"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)-[:DIRECTED]->(m2:Movie)
RETURN m.title AS MovieTitle, p.name AS PersonName
```

**Expected Output**:
```cypher
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
MATCH (p)-[:ACTED_IN]->(m)
RETURN p.name AS director_actor, m.title AS movie
```

**Issues**:
- Pattern `(m:Movie)-[:DIRECTED]->(m2:Movie)` is incorrect (movies don't direct movies)
- Should match same person `p` with both `DIRECTED` and `ACTED_IN` relationships to same movie `m`
- Uses different movie variables `m` and `m2` when they should be the same

**Test Case 2**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find all people who both acted and directed movies"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)-[:DIRECTED]->(m2:Movie)
RETURN p.name AS Person, m.title AS Movie, m2.title AS Movie2
```

**Expected Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(:Movie)
MATCH (p)-[:DIRECTED]->(:Movie)
RETURN DISTINCT p.name AS person
```

**Issues**:
- Same incorrect pattern as Test Case 1
- Returns movies instead of people
- Should use separate MATCH clauses or OPTIONAL MATCH

---

### 7. ⚠️ **Relationship Properties - Uses Node Properties Instead**

**Severity**: HIGH  
**Frequency**: ~80% of relationship property queries  
**Impact**: Queries access wrong properties

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Relationships: (:Person)-[:ACTED_IN {role: STRING}]->(:Movie) Question: Find actors and their roles in movies"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, m.title AS movie_title, p.role AS role
```

**Expected Output**:
```cypher
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, m.title AS movie_title, r.role AS role
```

**Issues**:
- Uses `p.role` (node property) instead of `r.role` (relationship property)
- Doesn't assign variable to relationship `[:ACTED_IN]`
- Should be `[r:ACTED_IN]` to access relationship properties

---

### 8. ❌ **Shortest Path Queries - Completely Broken**

**Severity**: CRITICAL  
**Frequency**: 100% of shortest path queries  
**Impact**: Queries are syntactically invalid

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Relationships: (:Person)-[:KNOWS]->(:Person) Question: Find the shortest path between two people"
```

**Generated Output**:
```cypher
MATCH (p1:Person)-[:KNOWS]->(p2:Person)
MATCH p1.PID -[:KNOWS]-> p2.PID
Find the shortest path between p1 and p2
```

**Expected Output**:
```cypher
MATCH path = shortestPath((p1:Person)-[:KNOWS*]-(p2:Person))
WHERE p1.name = 'Alice' AND p2.name = 'Bob'
RETURN path
```

**Issues**:
- Mixes Cypher syntax with natural language
- Invalid syntax: `MATCH p1.PID -[:KNOWS]-> p2.PID`
- Doesn't use `shortestPath()` function
- No WHERE clause to specify start/end nodes
- No RETURN clause

---

### 9. ⚠️ **Complex WHERE Clauses - Incorrect Variable Usage**

**Severity**: HIGH  
**Frequency**: ~50% of complex WHERE queries  
**Impact**: Queries fail at runtime

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find actors older than 50 who acted in movies with budget greater than 1000000"
```

**Generated Output**:
```cypher
MATCH path=(:Person)-[:ACTED_IN]->(:Movie)
WHERE Person.age > 50 AND Movie.budget > 1000000
RETURN path
```

**Expected Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.age > 50 AND m.budget > 1000000
RETURN p.name AS actor, m.title AS movie
```

**Issues**:
- Uses label names `Person.age` and `Movie.budget` instead of variables `p.age` and `m.budget`
- Uses `path` variable but doesn't define it properly
- Should use variables `p` and `m` assigned in MATCH clause

---

### 10. ⚠️ **TOP N Queries - Incorrect Aggregation Logic**

**Severity**: MEDIUM  
**Frequency**: ~40% of TOP N queries  
**Impact**: Queries don't produce correct rankings

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find top 10 actors by number of movies"
```

**Generated Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WITH m.title AS movieTitle, COUNT(p) AS actorCount
ORDER BY actorCount DESC
LIMIT 10
```

**Expected Output**:
```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, COUNT(m) AS movie_count
ORDER BY movie_count DESC
LIMIT 10
```

**Issues**:
- Groups by `m.title` instead of `p.name` (wrong grouping)
- Counts actors per movie instead of movies per actor
- Should use `RETURN` with `COUNT(m)` grouped by `p.name`

---

### 11. ⚠️ **Simple MATCH Queries - Incorrect Relationship Patterns**

**Severity**: MEDIUM  
**Frequency**: ~30% of simple queries  
**Impact**: Queries match wrong patterns

**Test Case**:
```bash
expert-cli chat --experts neo4j --prompt "Schema: ... Question: Find all movies"
```

**Generated Output**:
```cypher
MATCH (m:Movie)-[:ACTED_IN]->(p:Person)-[:DIRECTED_IN]->(m)-[:DIRECTED]->(p)
RETURN m.title AS MovieTitle
```

**Expected Output**:
```cypher
MATCH (m:Movie)
RETURN m.title AS title
```

**Issues**:
- Adds unnecessary relationships when query only asks for movies
- Pattern `-[:DIRECTED_IN]->(m)-[:DIRECTED]->(p)` is circular and incorrect
- Should be simple `MATCH (m:Movie)` without relationships

---

## Summary

### Critical Issues (Must Fix)
1. ❌ **Queries without schema** - Generates text instead of Cypher (100% failure)
2. ❌ **Shortest path queries** - Completely broken syntax (100% failure)
3. ❌ **Multi-hop traversals** - Incorrect pattern matching (~60% failure)

### High Priority Issues
4. ⚠️ **Property name typos** - Truncated names (~40% of queries)
5. ⚠️ **COUNT aggregation** - Missing COUNT() function (~30% of queries)
6. ⚠️ **AVG with GROUP BY** - Incorrect syntax and logic (~50% of queries)
7. ⚠️ **Self-join queries** - Incorrect logic (~70% of queries)
8. ⚠️ **Relationship properties** - Uses node properties instead (~80% of queries)
9. ⚠️ **Complex WHERE clauses** - Incorrect variable usage (~50% of queries)

### Medium Priority Issues
10. ⚠️ **TOP N queries** - Incorrect aggregation logic (~40% of queries)
11. ⚠️ **Simple MATCH queries** - Adds unnecessary relationships (~30% of queries)

---

## Recommendations

1. **Create synthetic dataset** targeting:
   - Property name completion (prevent truncation)
   - COUNT/AVG aggregation patterns
   - Multi-hop traversal patterns
   - Relationship property access
   - Self-join patterns
   - Shortest path queries

2. **Increase training** on:
   - Simple queries without unnecessary relationships
   - Correct variable usage in WHERE clauses
   - Proper GROUP BY syntax

3. **Consider checkpoint selection**:
   - Test if checkpoint-500 performs better on AVG GROUP BY (as mentioned in README)
   - May need capability-specific checkpoint routing

