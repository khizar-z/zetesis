const FULL_GRAPH_MAX_SEMANTIC_EDGES = 18000;
const NEIGHBORHOOD_MAX_SEMANTIC_EDGES = 12000;
const FULL_GRAPH_MAX_SEMANTIC_EDGES_PER_NODE = 18;
const NEIGHBORHOOD_MAX_SEMANTIC_EDGES_PER_NODE = 12;

function getSemanticEdgeBudget({ nodeCount, focusedSlug }) {
  if (focusedSlug) {
    return {
      maxSemanticEdges: Math.min(
        NEIGHBORHOOD_MAX_SEMANTIC_EDGES,
        Math.max(7000, Math.round(nodeCount * 6)),
      ),
      maxSemanticEdgesPerNode: NEIGHBORHOOD_MAX_SEMANTIC_EDGES_PER_NODE,
    };
  }

  return {
    maxSemanticEdges: Math.min(
      FULL_GRAPH_MAX_SEMANTIC_EDGES,
      Math.max(10000, Math.round(nodeCount * 12)),
    ),
    maxSemanticEdgesPerNode: FULL_GRAPH_MAX_SEMANTIC_EDGES_PER_NODE,
  };
}

function collectFocusedExplicitNeighbors(explicitEdges, focusedSlug) {
  if (!focusedSlug) {
    return new Set();
  }

  const neighbors = new Set();
  for (const edge of explicitEdges) {
    if (edge.source === focusedSlug) {
      neighbors.add(edge.target);
    } else if (edge.target === focusedSlug) {
      neighbors.add(edge.source);
    }
  }

  return neighbors;
}

function scoreSemanticEdge(edge, focusedSlug, focusedExplicitNeighbors) {
  let score = Number(edge.similarity) || 0;
  if (!focusedSlug) {
    return score;
  }

  const touchesFocus = edge.source === focusedSlug || edge.target === focusedSlug;
  const touchesFocusedNeighbor =
    focusedExplicitNeighbors.has(edge.source) || focusedExplicitNeighbors.has(edge.target);

  if (touchesFocus) {
    score += 2;
  } else if (touchesFocusedNeighbor) {
    score += 0.6;
  }

  return score;
}

function getNodeSemanticCap(slug, focusedSlug, perNodeCap) {
  if (focusedSlug && slug === focusedSlug) {
    return perNodeCap * 2;
  }
  return perNodeCap;
}

export function shouldCondenseSemanticEdges({ semanticEdgeCount, nodeCount, focusedSlug }) {
  return semanticEdgeCount > getSemanticEdgeBudget({ nodeCount, focusedSlug }).maxSemanticEdges;
}

export function selectRenderableSemanticEdges({
  explicitEdges,
  semanticEdges,
  focusedSlug,
  nodeCount,
}) {
  const { maxSemanticEdges, maxSemanticEdgesPerNode } = getSemanticEdgeBudget({
    nodeCount,
    focusedSlug,
  });

  if (semanticEdges.length <= maxSemanticEdges) {
    return semanticEdges;
  }

  const focusedExplicitNeighbors = collectFocusedExplicitNeighbors(explicitEdges, focusedSlug);
  const rankedEdges = [...semanticEdges].sort((left, right) => {
    const scoreDelta =
      scoreSemanticEdge(right, focusedSlug, focusedExplicitNeighbors) -
      scoreSemanticEdge(left, focusedSlug, focusedExplicitNeighbors);
    if (scoreDelta !== 0) {
      return scoreDelta;
    }

    return (Number(right.similarity) || 0) - (Number(left.similarity) || 0);
  });

  const nodeSemanticCounts = new Map();
  const selectedEdges = [];

  for (const edge of rankedEdges) {
    const sourceCount = nodeSemanticCounts.get(edge.source) || 0;
    const targetCount = nodeSemanticCounts.get(edge.target) || 0;
    const sourceCap = getNodeSemanticCap(edge.source, focusedSlug, maxSemanticEdgesPerNode);
    const targetCap = getNodeSemanticCap(edge.target, focusedSlug, maxSemanticEdgesPerNode);

    if (sourceCount >= sourceCap || targetCount >= targetCap) {
      continue;
    }

    selectedEdges.push(edge);
    nodeSemanticCounts.set(edge.source, sourceCount + 1);
    nodeSemanticCounts.set(edge.target, targetCount + 1);

    if (selectedEdges.length >= maxSemanticEdges) {
      break;
    }
  }

  return selectedEdges;
}
