# Geometric Semantic Recursion

Spatial semantic mapping of agent conversations in N-dimensional space — a framework for flexible agentic governance through geometric context.

## The Idea

Agent conversations have structure. A developer asking about a memory leak and a designer asking about colour contrast are *semantically distant* — and that distance is measurable. Embed the exchanges, project them into 3D space, and the topology reveals natural clusters: intent regions, domain neighbourhoods, context zones.

This repository explores using that geometry as a governance primitive:
- Route messages to the right agent based on semantic proximity to known regions
- Detect context drift when a conversation moves unexpectedly across the map
- Visualise agent behaviour across sessions as a living point cloud

## Demo: Semantic 3D Point Cloud

Builds an interactive Three.js viewer from a conversation database. Each point is a user/assistant exchange, embedded and projected to 3D. Points are coloured by context label (persona, category, or any string tag).

### Setup

```bash
pip install -r requirements.txt
```

Requires an OpenAI API key (for `text-embedding-3-small`):

```bash
export OPENAI_API_KEY=sk-...
```

### Run

```bash
python pointcloud_build.py --db path/to/your.db
```

Defaults: reads `data/datum-ui.db`, outputs `pointcloud.html`. Open the HTML file in any browser.

```
options:
  --db      Path to SQLite database (default: data/datum-ui.db)
  --limit   Max message pairs to embed (default: 1000)
  --out     Output HTML path (default: pointcloud.html)
  --cache   Embedding cache path (default: data/embeddings_cache.npy)
  --no-cache  Ignore cached embeddings and re-embed
```

### Database Schema

```sql
CREATE TABLE conversations (
    id      TEXT PRIMARY KEY,
    persona TEXT,   -- label used for colouring (e.g. 'developer', 'researcher')
    title   TEXT
);

CREATE TABLE messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    role            TEXT,  -- 'user' | 'assistant'
    content         TEXT,
    created_at      TEXT
);
```

The `persona` column drives the colour coding. Swap in any string tag — the viewer auto-assigns colours to unknown labels.

### Controls

| Action | Effect |
|--------|--------|
| Drag | Rotate |
| Scroll | Zoom |
| Hover | Show message preview |

Auto-rotates when idle.

### Cost

`text-embedding-3-small` costs ~$0.02 per million tokens. 1,000 message pairs (400 chars each) ≈ 100K tokens ≈ **$0.002**. Embeddings are cached to `data/embeddings_cache.npy` so re-runs are free.

## Governance Applications

The geometry is the insight. Clusters in the point cloud correspond to intent regions. Applications:

- **Routing**: classify new messages by proximity to known cluster centroids
- **Drift detection**: flag when a session trajectory crosses cluster boundaries unexpectedly
- **Access control**: define semantic "zones" where certain tools or escalations are allowed
- **Audit**: replay a session as a path through semantic space — visible, inspectable governance

## Built With

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) — `text-embedding-3-small`
- [UMAP](https://umap-learn.readthedocs.io/) — dimensionality reduction
- [Three.js](https://threejs.org/) — 3D rendering
- [Datum](https://github.com/mrodger) — the multi-persona agent platform this was built on
