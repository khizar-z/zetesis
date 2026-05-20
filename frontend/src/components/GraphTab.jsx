import GraphCanvas from "./GraphCanvas";
import {
  DEFAULT_SUBDISCIPLINE_COLOR,
  SUBDISCIPLINE_COLORS,
  SUBDISCIPLINE_LEGEND,
  getSubdisciplineColor,
} from "../subdisciplineConfig";
import { shouldCondenseSemanticEdges } from "../graphVisualization";

function pluralize(count, singular, plural) {
  return count === 1 ? singular : plural;
}

function formatViewTitle(view, selectedNode) {
  if (view.kind === "neighborhood" && selectedNode) {
    return `Neighborhood around ${selectedNode.title}`;
  }

  return "SEP Concept Graph";
}

function GraphLegend() {
  return (
    <div className="graph-legend">
      <div className="graph-legend__section">
        <h3>Subdisciplines</h3>
        <div className="graph-legend__colors">
          {SUBDISCIPLINE_LEGEND.map((label) => (
            <div key={label} className="graph-legend__item">
              <span
                className="graph-legend__swatch"
                style={{ backgroundColor: SUBDISCIPLINE_COLORS[label] }}
              />
              <span>{label}</span>
            </div>
          ))}
          <div className="graph-legend__item">
            <span
              className="graph-legend__swatch"
              style={{ backgroundColor: DEFAULT_SUBDISCIPLINE_COLOR }}
            />
            <span>Unclassified</span>
          </div>
        </div>
      </div>

      <div className="graph-legend__section">
        <h3>Edge Types</h3>
        <div className="graph-legend__lines">
          <div className="graph-legend__line-item">
            <span className="graph-legend__line graph-legend__line--explicit" />
            <span>Explicit SEP links</span>
          </div>
          <div className="graph-legend__line-item">
            <span className="graph-legend__line graph-legend__line--semantic" />
            <span>Semantic similarity</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function GraphInfoPanel({ selectedNode, onSearchEntry }) {
  if (!selectedNode) {
    return (
      <aside className="graph-info">
        <p className="graph-info__eyebrow">Entry Focus</p>
        <h2>Pick an SEP entry to open its neighborhood.</h2>
        <p className="graph-info__copy">
          Search for an entry first, or open one from the results list. Once a neighborhood is
          loaded, clicking any node recenters the graph around that entry's one-hop connections.
        </p>
      </aside>
    );
  }

  const chipColor = getSubdisciplineColor(selectedNode.subdiscipline);

  return (
    <aside className="graph-info">
      <p className="graph-info__eyebrow">Entry Focus</p>
      <h2>{selectedNode.title}</h2>

      <div className="graph-info__chips">
        <span className="graph-info__chip" style={{ backgroundColor: `${chipColor}22`, color: chipColor }}>
          {selectedNode.subdiscipline || "Unclassified"}
        </span>
        <span className="graph-info__chip graph-info__chip--muted">
          {selectedNode.degree} {pluralize(selectedNode.degree, "connection", "connections")}
        </span>
      </div>

      <p className="graph-info__copy">
        This panel updates when you click a node. Use it as a bridge back into passage search if
        you want to read the entry itself rather than just its graph neighborhood.
      </p>

      <button
        className="graph-info__button"
        type="button"
        onClick={() => onSearchEntry(selectedNode.title)}
      >
        Search this entry
      </button>
    </aside>
  );
}

export default function GraphTab({
  graphData,
  graphStatus,
  graphErrorMessage,
  graphView,
  selectedGraphNode,
  onBack,
  onBackToSearch,
  onGraphNodeSelect,
  onSearchEntry,
  onRetryGraph,
}) {
  const hasGraph = Boolean(graphData);
  const isLoading = graphStatus === "loading";
  const showLoadingState = isLoading && !hasGraph;
  const showInitialState = graphStatus === "idle" && !hasGraph;
  const showError = graphStatus === "error" && !hasGraph;

  const nodeCount = graphData?.nodes.length || 0;
  const explicitEdgeCount = graphData?.explicit_edges.length || 0;
  const semanticEdgeCount = graphData?.semantic_edges.length || 0;
  const totalEdgeCount = explicitEdgeCount + semanticEdgeCount;
  const isDenseGraph = hasGraph
    ? shouldCondenseSemanticEdges({
        semanticEdgeCount,
        nodeCount,
        focusedSlug: graphView.centeredSlug,
      })
    : false;

  return (
    <section className="graph-tab">
      <div className="graph-tab__header">
        <div>
          <p className="graph-tab__eyebrow">Second Interface</p>
          <h2>{formatViewTitle(graphView, selectedGraphNode)}</h2>
          <p className="graph-tab__copy">
            Explore a single SEP entry's local neighborhood through direct cross-links and strict
            semantic similarity.
          </p>
        </div>

        <div className="graph-tab__actions">
          <button className="graph-tab__back" type="button" onClick={onBack}>
            Back
          </button>
          <button className="graph-tab__back" type="button" onClick={onBackToSearch}>
            Back to search
          </button>
        </div>
      </div>

      {showInitialState ? (
        <div className="status-card">
          <div>
            <h2>No entry selected yet</h2>
            <p>
              Search SEP first, then open an entry from the results list to render its one-hop
              neighborhood.
            </p>
          </div>
        </div>
      ) : null}

      {showLoadingState ? (
        <div className="status-card" role="status" aria-live="polite">
          <div className="status-card__spinner" />
          <div>
            <h2>Loading SEP neighborhood</h2>
            <p>
              {selectedGraphNode
                ? `Building the one-hop graph around ${selectedGraphNode.title}.`
                : "Building the entry neighborhood graph."}
            </p>
          </div>
        </div>
      ) : null}

      {showError ? (
        <div className="status-card status-card--error" role="alert">
          <div>
            <h2>Graph unavailable</h2>
            <p>{graphErrorMessage}</p>
          </div>
          <button className="graph-tab__retry" type="button" onClick={onRetryGraph}>
            Try again
          </button>
        </div>
      ) : null}

      {hasGraph ? (
        <>
          <div className="graph-summary">
            <p>
              {nodeCount} {pluralize(nodeCount, "entry", "entries")} · {explicitEdgeCount} explicit{" "}
              {pluralize(explicitEdgeCount, "edge", "edges")} · {semanticEdgeCount} semantic{" "}
              {pluralize(semanticEdgeCount, "edge", "edges")}
            </p>
            <p>{totalEdgeCount} total graph relationships loaded from the API.</p>
            {isDenseGraph ? (
              <p className="graph-summary__note">
                Dense views emphasize the strongest semantic ties so the graph stays legible and
                clickable.
              </p>
            ) : null}
          </div>

          <div className="graph-workspace">
            <div className="graph-stage">
              <GraphCanvas
                data={graphData}
                focusedSlug={graphView.centeredSlug}
                onNodeSelect={onGraphNodeSelect}
              />

              {isLoading ? (
                <div className="graph-stage__overlay" role="status" aria-live="polite">
                  <div className="status-card__spinner" />
                  <p>
                    {graphView.kind === "neighborhood" && selectedGraphNode
                      ? `Centering on ${selectedGraphNode.title}...`
                      : "Loading SEP neighborhood..."}
                  </p>
                </div>
              ) : null}
            </div>

            <GraphInfoPanel selectedNode={selectedGraphNode} onSearchEntry={onSearchEntry} />
          </div>

          {graphStatus === "error" && graphErrorMessage ? (
            <div className="status-card status-card--error graph-tab__inline-error" role="alert">
              <h2>Graph request failed</h2>
              <p>{graphErrorMessage}</p>
            </div>
          ) : null}

          <GraphLegend />
        </>
      ) : null}
    </section>
  );
}
