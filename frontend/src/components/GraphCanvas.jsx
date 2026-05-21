import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

import { getSubdisciplineColor } from "../subdisciplineConfig";
import { selectRenderableSemanticEdges } from "../graphVisualization";

const GRAPH_HEIGHT_MIN = 520;
const GRAPH_HEIGHT_MAX = 760;
const MIN_NODE_RADIUS = 4.5;
const MAX_NODE_RADIUS = 13;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function buildNeighborMap(data) {
  const neighbors = new Map();

  function ensure(slug) {
    if (!neighbors.has(slug)) {
      neighbors.set(slug, new Set());
    }
    return neighbors.get(slug);
  }

  for (const node of data.nodes) {
    ensure(node.slug);
  }

  for (const edge of data.explicit_edges) {
    ensure(edge.source).add(edge.target);
    ensure(edge.target).add(edge.source);
  }

  for (const edge of data.semantic_edges) {
    ensure(edge.source).add(edge.target);
    ensure(edge.target).add(edge.source);
  }

  return neighbors;
}

function createNodeRadiusScale(nodes) {
  const degrees = nodes.map((node) => Number(node.degree) || 0);
  const maxDegree = Math.max(...degrees, 1);

  return d3.scaleSqrt().domain([0, maxDegree]).range([MIN_NODE_RADIUS, MAX_NODE_RADIUS]);
}

function createEdgeWeightScale(edges) {
  const maxWeight = Math.max(...edges.map((edge) => Number(edge.weight) || 1), 1);
  return d3.scaleLinear().domain([1, maxWeight]).range([0.8, 3.2]);
}

function createSemanticOpacityScale(edges) {
  const similarities = edges.map((edge) => Number(edge.similarity) || 0);
  const minSimilarity = Math.min(...similarities, 0.5);
  const maxSimilarity = Math.max(...similarities, 0.9);

  if (minSimilarity === maxSimilarity) {
    return () => 0.18;
  }

  return d3.scaleLinear().domain([minSimilarity, maxSimilarity]).range([0.05, 0.2]);
}

function getChargeStrength(nodeCount) {
  if (nodeCount <= 24) {
    return -90;
  }

  if (nodeCount <= 48) {
    return -72;
  }

  if (nodeCount <= 96) {
    return -52;
  }

  if (nodeCount > 1500) {
    return -10;
  }

  if (nodeCount > 500) {
    return -16;
  }

  return -26;
}

function getLinkDistance(nodeCount, weight) {
  const normalizedWeight = Math.min(Number(weight) || 1, 6);

  if (nodeCount <= 24) {
    return 162 - normalizedWeight * 10;
  }

  if (nodeCount <= 48) {
    return 142 - normalizedWeight * 9;
  }

  if (nodeCount <= 96) {
    return 118 - normalizedWeight * 8;
  }

  return 76 - normalizedWeight * 6;
}

function getTickCount(nodeCount) {
  if (nodeCount > 1500) {
    return 160;
  }

  if (nodeCount > 800) {
    return 210;
  }

  return 260;
}

function fitNodesToViewport(nodes, viewport, nodeRadiusScale, focusNode) {
  if (!nodes.length) {
    return;
  }

  const padding = nodes.length <= 40 ? 74 : nodes.length <= 90 ? 64 : 54;
  const maxScale = nodes.length <= 24 ? 4.4 : nodes.length <= 48 ? 3.8 : nodes.length <= 96 ? 3.1 : 2.2;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const node of nodes) {
    const radius = nodeRadiusScale(node.degree) + 10;
    minX = Math.min(minX, node.x - radius);
    maxX = Math.max(maxX, node.x + radius);
    minY = Math.min(minY, node.y - radius);
    maxY = Math.max(maxY, node.y + radius);
  }

  const boxWidth = Math.max(maxX - minX, 1);
  const boxHeight = Math.max(maxY - minY, 1);
  const availableWidth = Math.max(viewport.width - padding * 2, 1);
  const availableHeight = Math.max(viewport.height - padding * 2, 1);
  const scale = clamp(Math.min(availableWidth / boxWidth, availableHeight / boxHeight), 1, maxScale);
  const sourceCenterX = focusNode ? focusNode.x : (minX + maxX) / 2;
  const sourceCenterY = focusNode ? focusNode.y : (minY + maxY) / 2;
  const targetCenterX = viewport.width / 2;
  const targetCenterY = viewport.height / 2;

  const transformed = nodes.map((node) => ({
    node,
    x: (node.x - sourceCenterX) * scale + targetCenterX,
    y: (node.y - sourceCenterY) * scale + targetCenterY,
  }));

  let transformedMinX = Infinity;
  let transformedMaxX = -Infinity;
  let transformedMinY = Infinity;
  let transformedMaxY = -Infinity;

  for (const item of transformed) {
    const radius = nodeRadiusScale(item.node.degree) + 10;
    transformedMinX = Math.min(transformedMinX, item.x - radius);
    transformedMaxX = Math.max(transformedMaxX, item.x + radius);
    transformedMinY = Math.min(transformedMinY, item.y - radius);
    transformedMaxY = Math.max(transformedMaxY, item.y + radius);
  }

  let translateX = 0;
  if (transformedMinX < padding) {
    translateX = padding - transformedMinX;
  }
  if (transformedMaxX + translateX > viewport.width - padding) {
    translateX -= transformedMaxX + translateX - (viewport.width - padding);
  }

  let translateY = 0;
  if (transformedMinY < padding) {
    translateY = padding - transformedMinY;
  }
  if (transformedMaxY + translateY > viewport.height - padding) {
    translateY -= transformedMaxY + translateY - (viewport.height - padding);
  }

  for (const item of transformed) {
    item.node.x = item.x + translateX;
    item.node.y = item.y + translateY;
    if ("fx" in item.node && item.node.fx != null) {
      item.node.fx = item.node.x;
    }
    if ("fy" in item.node && item.node.fy != null) {
      item.node.fy = item.node.y;
    }
  }
}

function drawRoundedRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

export default function GraphCanvas({ data, focusedSlug, onNodeSelect }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const graphRef = useRef(null);
  const drawRef = useRef(() => {});
  const hoveredSlugRef = useRef(null);
  const [hoveredSlug, setHoveredSlug] = useState(null);
  const [viewport, setViewport] = useState({
    width: 960,
    height: 620,
  });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    function updateViewport(nextWidth) {
      const width = Math.max(320, Math.floor(nextWidth));
      const height = clamp(Math.round(width * 0.68), GRAPH_HEIGHT_MIN, GRAPH_HEIGHT_MAX);
      setViewport((current) => {
        if (current.width === width && current.height === height) {
          return current;
        }

        return { width, height };
      });
    }

    updateViewport(container.getBoundingClientRect().width);
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      updateViewport(entry.contentRect.width);
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    hoveredSlugRef.current = hoveredSlug;
    drawRef.current();
  }, [hoveredSlug]);

  useEffect(() => {
    setHoveredSlug(null);
  }, [data, focusedSlug]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || viewport.width <= 0 || viewport.height <= 0) {
      return undefined;
    }

    const devicePixelRatio = window.devicePixelRatio || 1;
    canvas.width = viewport.width * devicePixelRatio;
    canvas.height = viewport.height * devicePixelRatio;
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;

    const context = canvas.getContext("2d");
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

    const renderableSemanticEdges = selectRenderableSemanticEdges({
      explicitEdges: data.explicit_edges,
      semanticEdges: data.semantic_edges,
      focusedSlug,
      nodeCount: data.nodes.length,
    });
    const nodeRadiusScale = createNodeRadiusScale(data.nodes);
    const explicitEdgeWidth = createEdgeWeightScale(data.explicit_edges);
    const semanticOpacity = createSemanticOpacityScale(renderableSemanticEdges);
    const neighborsBySlug = buildNeighborMap({
      ...data,
      semantic_edges: renderableSemanticEdges,
    });

    const simulationNodes = data.nodes.map((node, index) => ({
      ...node,
      index,
      x: (index % 24) * 24 + (index % 3) * 5,
      y: Math.floor(index / 24) * 24 + (index % 5) * 3,
    }));

    const nodesBySlug = new Map(simulationNodes.map((node) => [node.slug, node]));
    const explicitLinks = data.explicit_edges
      .map((edge) => {
        const sourceNode = nodesBySlug.get(edge.source);
        const targetNode = nodesBySlug.get(edge.target);
        if (!sourceNode || !targetNode) {
          return null;
        }

        return {
          ...edge,
          source: sourceNode,
          target: targetNode,
        };
      })
      .filter(Boolean);

    const semanticLinks = renderableSemanticEdges
      .map((edge) => {
        const sourceNode = nodesBySlug.get(edge.source);
        const targetNode = nodesBySlug.get(edge.target);
        if (!sourceNode || !targetNode) {
          return null;
        }

        return {
          ...edge,
          source: sourceNode,
          target: targetNode,
        };
      })
      .filter(Boolean);
    const nodesInDrawOrder = [...simulationNodes].sort((left, right) => left.degree - right.degree);
    const focusNode = focusedSlug ? nodesBySlug.get(focusedSlug) : null;

    if (focusNode) {
      focusNode.fx = viewport.width / 2;
      focusNode.fy = viewport.height / 2;
      focusNode.x = viewport.width / 2;
      focusNode.y = viewport.height / 2;
    }

    const simulation = d3
      .forceSimulation(simulationNodes)
      .force(
        "charge",
        d3.forceManyBody().strength(getChargeStrength(data.nodes.length)),
      )
      .force("center", d3.forceCenter(viewport.width / 2, viewport.height / 2))
      .force(
        "collision",
        d3.forceCollide().radius((node) => nodeRadiusScale(node.degree) + 4),
      )
      .force(
        "x",
        d3
          .forceX(viewport.width / 2)
          .strength(focusNode ? 0.03 : 0.016),
      )
      .force(
        "y",
        d3
          .forceY(viewport.height / 2)
          .strength(focusNode ? 0.03 : 0.016),
      )
      .force(
        "link",
        explicitLinks.length > 0
          ? d3
              .forceLink(explicitLinks)
              .id((node) => node.slug)
              .distance((link) => getLinkDistance(data.nodes.length, link.weight))
              .strength(0.14)
          : null,
      )
      .alpha(1)
      .alphaDecay(data.nodes.length > 1500 ? 0.12 : data.nodes.length > 500 ? 0.08 : 0.06)
      .velocityDecay(0.28);

    for (let tick = 0; tick < getTickCount(data.nodes.length); tick += 1) {
      simulation.tick();
    }
    simulation.stop();
    fitNodesToViewport(simulationNodes, viewport, nodeRadiusScale, focusNode);

    graphRef.current = {
      simulation,
      nodes: simulationNodes,
      nodesBySlug,
      nodesInDrawOrder,
      neighborsBySlug,
      nodeRadiusScale,
      explicitLinks,
      semanticLinks,
    };

    function draw() {
      const graph = graphRef.current;
      if (!graph) {
        return;
      }

      const currentHoveredSlug = hoveredSlugRef.current;
      const highlightedSlugs = currentHoveredSlug
        ? new Set([currentHoveredSlug, ...(graph.neighborsBySlug.get(currentHoveredSlug) || [])])
        : null;

      context.clearRect(0, 0, viewport.width, viewport.height);
      context.fillStyle = "rgba(255, 252, 247, 0.92)";
      context.fillRect(0, 0, viewport.width, viewport.height);

      context.save();
      context.setLineDash([5, 7]);
      for (const edge of graph.semanticLinks) {
        const source = edge.source;
        const target = edge.target;
        const isHighlighted =
          highlightedSlugs === null ||
          highlightedSlugs.has(source.slug) ||
          highlightedSlugs.has(target.slug);

        const opacity = semanticOpacity(edge.similarity) * (isHighlighted ? 1 : 0.12);
        if (opacity <= 0.008) {
          continue;
        }

        context.beginPath();
        context.strokeStyle = `rgba(73, 106, 140, ${opacity})`;
        context.lineWidth = isHighlighted ? 0.9 : 0.55;
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.stroke();
      }
      context.restore();

      for (const edge of graph.explicitLinks) {
        const source = edge.source;
        const target = edge.target;
        const isHighlighted =
          highlightedSlugs === null ||
          highlightedSlugs.has(source.slug) ||
          highlightedSlugs.has(target.slug);

        context.beginPath();
        context.strokeStyle = isHighlighted
          ? "rgba(111, 81, 45, 0.48)"
          : "rgba(111, 81, 45, 0.09)";
        context.lineWidth = explicitEdgeWidth(edge.weight) * (isHighlighted ? 1 : 0.72);
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.stroke();
      }

      for (const node of graph.nodesInDrawOrder) {
        const isHovered = node.slug === currentHoveredSlug;
        const isNeighbor = currentHoveredSlug
          ? graph.neighborsBySlug.get(currentHoveredSlug)?.has(node.slug)
          : false;
        const isHighlighted = highlightedSlugs === null || isHovered || isNeighbor;
        const radius = nodeRadiusScale(node.degree);

        context.beginPath();
        context.globalAlpha = isHighlighted ? 1 : 0.18;
        context.fillStyle = getSubdisciplineColor(node.subdiscipline);
        context.strokeStyle = isHovered ? "rgba(36, 26, 16, 0.92)" : "rgba(255, 253, 248, 0.88)";
        context.lineWidth = isHovered ? 2.4 : 1.1;
        context.arc(node.x, node.y, radius, 0, Math.PI * 2);
        context.fill();
        context.stroke();

        if (focusedSlug && node.slug === focusedSlug) {
          context.beginPath();
          context.globalAlpha = 1;
          context.strokeStyle = "rgba(36, 26, 16, 0.36)";
          context.lineWidth = 6;
          context.arc(node.x, node.y, radius + 4, 0, Math.PI * 2);
          context.stroke();
        }
      }
      context.globalAlpha = 1;

      if (!currentHoveredSlug) {
        return;
      }

      const hoveredNode = graph.nodesBySlug.get(currentHoveredSlug);
      if (!hoveredNode) {
        return;
      }

      const label = hoveredNode.title;
      context.font = '600 13px "Source Sans 3", sans-serif';
      const textWidth = context.measureText(label).width;
      const boxWidth = textWidth + 16;
      const boxHeight = 30;
      const offsetX = hoveredNode.x + 16;
      const offsetY = hoveredNode.y - 38;
      const labelX = clamp(offsetX, 10, viewport.width - boxWidth - 10);
      const labelY = clamp(offsetY, 10, viewport.height - boxHeight - 10);

      context.fillStyle = "rgba(255, 252, 247, 0.96)";
      context.strokeStyle = "rgba(111, 81, 45, 0.2)";
      context.lineWidth = 1;
      drawRoundedRect(context, labelX, labelY, boxWidth, boxHeight, 10);
      context.fill();
      context.stroke();

      context.fillStyle = "rgba(36, 26, 16, 0.96)";
      context.textBaseline = "middle";
      context.fillText(label, labelX + 8, labelY + boxHeight / 2);
    }

    drawRef.current = draw;
    draw();

    return () => {
      drawRef.current = () => {};
      graphRef.current = null;
      simulation.stop();
    };
  }, [data, focusedSlug, viewport.height, viewport.width]);

  function findNodeForPointer(event) {
    const graph = graphRef.current;
    const canvas = canvasRef.current;
    if (!graph || !canvas) {
      return null;
    }

    const bounds = canvas.getBoundingClientRect();
    const pointerX = event.clientX - bounds.left;
    const pointerY = event.clientY - bounds.top;
    const nearest = graph.simulation.find(pointerX, pointerY, 20);
    if (!nearest) {
      return null;
    }

    const radius = graph.nodeRadiusScale(nearest.degree) + 6;
    const dx = nearest.x - pointerX;
    const dy = nearest.y - pointerY;
    return Math.hypot(dx, dy) <= radius ? nearest : null;
  }

  function handlePointerMove(event) {
    const hoveredNode = findNodeForPointer(event);
    const nextHoveredSlug = hoveredNode ? hoveredNode.slug : null;
    event.currentTarget.style.cursor = hoveredNode ? "pointer" : "default";

    setHoveredSlug((current) => (current === nextHoveredSlug ? current : nextHoveredSlug));
  }

  function handlePointerLeave(event) {
    event.currentTarget.style.cursor = "default";
    setHoveredSlug(null);
  }

  function handleClick(event) {
    const selectedNode = findNodeForPointer(event);
    if (!selectedNode) {
      return;
    }

    onNodeSelect({
      slug: selectedNode.slug,
      title: selectedNode.title,
      subdiscipline: selectedNode.subdiscipline,
      degree: selectedNode.degree,
      intro_text: selectedNode.intro_text,
    });
  }

  return (
    <div className="graph-canvas-shell" ref={containerRef}>
      <canvas
        ref={canvasRef}
        className="graph-canvas"
        onMouseMove={handlePointerMove}
        onMouseLeave={handlePointerLeave}
        onClick={handleClick}
      />
    </div>
  );
}
