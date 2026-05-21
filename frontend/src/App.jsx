import { startTransition, useEffect, useRef, useState } from "react";

import EntryGroup from "./components/EntryGroup";
import GraphTab from "./components/GraphTab";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);
const RESULTS_PAGE_SIZE = 7;

const EXAMPLE_QUERIES = [
  "relationship between free will and moral responsibility",
  "details of the nonidentity problem",
  "what is truth",
];

function buildSearchUrl(query, limit = RESULTS_PAGE_SIZE) {
  return `${API_BASE_URL}/search?q=${encodeURIComponent(query)}&limit=${limit}`;
}

function buildNeighborhoodUrl(slug, hops = 1) {
  return `${API_BASE_URL}/graph/neighborhood?slug=${encodeURIComponent(slug)}&hops=${hops}`;
}

function getEntryPageUrl(url) {
  return url.split("#")[0];
}

function groupResultsByEntry(results) {
  const groups = [];
  const groupsByPageUrl = new Map();

  for (let index = 0; index < results.length; index += 1) {
    const result = results[index];
    const pageUrl = getEntryPageUrl(result.url);

    if (!groupsByPageUrl.has(pageUrl)) {
      const group = {
        pageUrl,
        entrySlug: result.entry_slug,
        entryTitle: result.entry_title,
        startIndex: index,
        results: [],
      };
      groupsByPageUrl.set(pageUrl, group);
      groups.push(group);
    }

    groupsByPageUrl.get(pageUrl).results.push(result);
  }

  return groups;
}

function isGraphPayload(payload) {
  return (
    payload &&
    Array.isArray(payload.nodes) &&
    Array.isArray(payload.explicit_edges) &&
    Array.isArray(payload.semantic_edges)
  );
}

function parseRouteFromPath(pathname) {
  const trimmed = pathname.replace(/\/+$/, "") || "/";

  if (trimmed === "/") {
    return { page: "search" };
  }

  if (trimmed === "/graph") {
    return { page: "graph", slug: null };
  }

  const graphMatch = trimmed.match(/^\/graph\/([^/]+)$/);
  if (graphMatch) {
    return {
      page: "graph",
      slug: decodeURIComponent(graphMatch[1]),
    };
  }

  return { page: "search" };
}

function routeToPath(route) {
  if (route.page === "graph") {
    return route.slug ? `/graph/${encodeURIComponent(route.slug)}` : "/graph";
  }

  return "/";
}

function getHistoryDepth() {
  if (typeof window === "undefined") {
    return 0;
  }

  const depth = window.history.state?.depth;
  return typeof depth === "number" ? depth : 0;
}

export default function App() {
  const [route, setRoute] = useState(() => {
    if (typeof window === "undefined") {
      return { page: "search" };
    }
    return parseRouteFromPath(window.location.pathname);
  });

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [requestedResultLimit, setRequestedResultLimit] = useState(RESULTS_PAGE_SIZE);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [loadMoreErrorMessage, setLoadMoreErrorMessage] = useState("");

  const [graphData, setGraphData] = useState(null);
  const [graphStatus, setGraphStatus] = useState("idle");
  const [graphErrorMessage, setGraphErrorMessage] = useState("");
  const [graphView, setGraphView] = useState({
    kind: "empty",
    centeredSlug: null,
  });
  const [selectedGraphNode, setSelectedGraphNode] = useState(null);

  const searchAbortControllerRef = useRef(null);
  const graphAbortControllerRef = useRef(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const initialRoute = parseRouteFromPath(window.location.pathname);
    const currentState =
      window.history.state && typeof window.history.state === "object" ? window.history.state : {};
    const depth = typeof currentState.depth === "number" ? currentState.depth : 0;

    window.history.replaceState(
      {
        ...currentState,
        depth,
      },
      "",
      routeToPath(initialRoute),
    );

    setRoute(initialRoute);

    if (initialRoute.page === "graph" && initialRoute.slug) {
      void loadNeighborhood({ slug: initialRoute.slug });
    }

    function handlePopState() {
      const nextRoute = parseRouteFromPath(window.location.pathname);
      startTransition(() => {
        setRoute(nextRoute);
      });

      if (nextRoute.page === "graph" && nextRoute.slug) {
        void loadNeighborhood({ slug: nextRoute.slug });
      } else {
        graphAbortControllerRef.current?.abort();
      }
    }

    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  useEffect(() => {
    return () => {
      searchAbortControllerRef.current?.abort();
      graphAbortControllerRef.current?.abort();
    };
  }, []);

  function updateHistory(nextRoute, { replace = false } = {}) {
    if (typeof window === "undefined") {
      startTransition(() => {
        setRoute(nextRoute);
      });
      return;
    }

    const method = replace ? "replaceState" : "pushState";
    const nextDepth = replace ? getHistoryDepth() : getHistoryDepth() + 1;

    window.history[method](
      {
        ...(window.history.state || {}),
        depth: nextDepth,
      },
      "",
      routeToPath(nextRoute),
    );

    startTransition(() => {
      setRoute(nextRoute);
    });
  }

  async function runSearch(
    nextQuery,
    { limit = RESULTS_PAGE_SIZE, preserveResults = false } = {},
  ) {
    const cleanedQuery = nextQuery.trim();
    if (!cleanedQuery) {
      setStatus("error");
      setResults([]);
      setSubmittedQuery("");
      setErrorMessage("Enter a philosophical question or concept to search SEP.");
      setRequestedResultLimit(RESULTS_PAGE_SIZE);
      setLoadMoreErrorMessage("");
      setIsLoadingMore(false);
      return;
    }

    searchAbortControllerRef.current?.abort();
    const controller = new AbortController();
    searchAbortControllerRef.current = controller;

    const isIncrementalLoad = preserveResults && results.length > 0;

    if (isIncrementalLoad) {
      setIsLoadingMore(true);
    } else {
      setStatus("loading");
      setResults([]);
    }

    setErrorMessage("");
    setLoadMoreErrorMessage("");
    setSubmittedQuery(cleanedQuery);

    try {
      const response = await fetch(buildSearchUrl(cleanedQuery, limit), {
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          payload && typeof payload.detail === "string"
            ? payload.detail
            : "Search failed. Please try again.";
        throw new Error(detail);
      }

      startTransition(() => {
        setResults(Array.isArray(payload) ? payload : []);
        setStatus("success");
        setRequestedResultLimit(limit);
      });
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }

      if (isIncrementalLoad) {
        setLoadMoreErrorMessage(error.message || "Could not load more results.");
      } else {
        setResults([]);
        setStatus("error");
        setErrorMessage(error.message || "Search failed. Please try again.");
      }
    } finally {
      if (isIncrementalLoad) {
        setIsLoadingMore(false);
      }
    }
  }

  async function loadNeighborhood(node) {
    if (!node?.slug) {
      return;
    }

    graphAbortControllerRef.current?.abort();
    const controller = new AbortController();
    graphAbortControllerRef.current = controller;

    setGraphStatus("loading");
    setGraphErrorMessage("");
    setSelectedGraphNode((current) => ({
      slug: node.slug,
      title: node.title || current?.title || "",
      subdiscipline: node.subdiscipline || current?.subdiscipline || null,
      degree: node.degree || current?.degree || 0,
    }));
    setGraphView({
      kind: "neighborhood",
      centeredSlug: node.slug,
    });

    try {
      const response = await fetch(buildNeighborhoodUrl(node.slug, 1), {
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          payload && typeof payload.detail === "string"
            ? payload.detail
            : "Could not load this graph neighborhood.";
        throw new Error(detail);
      }

      if (!isGraphPayload(payload)) {
        throw new Error("The graph response was not in the expected format.");
      }

      const centeredNode = payload.nodes.find((candidate) => candidate.slug === node.slug) || {
        slug: node.slug,
        title: node.title || "",
        subdiscipline: node.subdiscipline || null,
        degree: node.degree || 0,
      };

      startTransition(() => {
        setGraphData(payload);
        setGraphView({
          kind: "neighborhood",
          centeredSlug: node.slug,
        });
        setSelectedGraphNode(centeredNode);
        setGraphStatus("success");
      });
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }

      setGraphStatus("error");
      setGraphErrorMessage(error.message || "Could not load this graph neighborhood.");
    }
  }

  function navigateToSearch({ replace = false } = {}) {
    updateHistory({ page: "search" }, { replace });
  }

  function handleReturnHome() {
    searchAbortControllerRef.current?.abort();
    graphAbortControllerRef.current?.abort();

    const emptyGraphView = {
      kind: "empty",
      centeredSlug: null,
    };

    startTransition(() => {
      setQuery("");
      setResults([]);
      setStatus("idle");
      setSubmittedQuery("");
      setErrorMessage("");
      setRequestedResultLimit(RESULTS_PAGE_SIZE);
      setIsLoadingMore(false);
      setLoadMoreErrorMessage("");
      setGraphData(null);
      setGraphStatus("idle");
      setGraphErrorMessage("");
      setGraphView(emptyGraphView);
      setSelectedGraphNode(null);
    });

    navigateToSearch({ replace: route.page === "search" });
  }

  function handleOpenGraphForEntry(entry) {
    if (!entry?.slug) {
      return;
    }

    const nextRoute = {
      page: "graph",
      slug: entry.slug,
    };

    const isSameRoute = route.page === "graph" && route.slug === entry.slug;
    if (!isSameRoute) {
      updateHistory(nextRoute);
    }

    void loadNeighborhood(entry);
  }

  function handleGraphBack() {
    if (typeof window === "undefined") {
      navigateToSearch({ replace: true });
      return;
    }

    if (getHistoryDepth() > 0) {
      window.history.back();
      return;
    }

    navigateToSearch({ replace: true });
  }

  function handleBackToSearchFromGraph() {
    navigateToSearch();
  }

  function handleSearchEntryFromGraph(entryTitle) {
    navigateToSearch();
    startTransition(() => {
      setQuery(entryTitle);
    });
    void runSearch(entryTitle);
  }

  function handleRetryGraph() {
    if (route.page === "graph" && route.slug) {
      void loadNeighborhood({
        slug: route.slug,
        title: selectedGraphNode?.title || "",
      });
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    await runSearch(query);
  }

  function handleExampleClick(example) {
    setQuery(example);
    void runSearch(example);
  }

  function handleLoadMore() {
    if (!submittedQuery || isLoadingMore) {
      return;
    }

    void runSearch(submittedQuery, {
      limit: requestedResultLimit + RESULTS_PAGE_SIZE,
      preserveResults: true,
    });
  }

  const showInitialState = status === "idle" && results.length === 0 && !errorMessage;
  const showEmptyState = status === "success" && results.length === 0;
  const showResults = results.length > 0;
  const groupedResults = groupResultsByEntry(results);
  const entryCount = groupedResults.length;
  const topResult = results[0] || null;
  const isCenteredHome = route.page === "search" && showInitialState;
  const shouldShowLoadMore = showResults && results.length === requestedResultLimit;

  return (
    <div className="page-shell">
      <div className="page-shell__glow page-shell__glow--left" />
      <div className="page-shell__glow page-shell__glow--right" />

      <main className={`layout${isCenteredHome ? " layout--home" : ""}`}>
        {route.page === "search" ? (
          <>
            <section className="hero">
              <h1>
                <button className="hero__brand" type="button" onClick={handleReturnHome}>
                  Zetesis
                </button>
              </h1>
              <p className="hero__copy">
                Search and navigate the Stanford Encyclopedia of Philosophy.
              </p>

              <form className="search-panel" onSubmit={handleSubmit}>
                <label className="search-panel__label" htmlFor="search-query">
                  Ask a philosophical question or look up a philosophical term
                </label>
                <div className="search-panel__controls">
                  <input
                    id="search-query"
                    name="search-query"
                    className="search-panel__input"
                    type="text"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="e.g. relationship between free will and moral responsibility"
                    maxLength={300}
                    autoComplete="off"
                  />
                  <button
                    className="search-panel__button"
                    type="submit"
                    disabled={status === "loading"}
                  >
                    {status === "loading" ? "Searching..." : "Search"}
                  </button>
                </div>
              </form>

              {showInitialState ? (
                <div className="example-list" aria-label="Example queries">
                  {EXAMPLE_QUERIES.map((example) => (
                    <button
                      key={example}
                      className="example-list__button"
                      type="button"
                      onClick={() => handleExampleClick(example)}
                    >
                      {example}
                    </button>
                  ))}
                </div>
              ) : null}
            </section>

            {!showInitialState ? (
              <section className="results-panel">
              {status === "loading" ? (
                <div className="status-card" role="status" aria-live="polite">
                  <div className="status-card__spinner" />
                  <div>
                    <h2>Searching SEP</h2>
                    <p>Finding the most relevant passages for “{submittedQuery}”.</p>
                  </div>
                </div>
              ) : null}

              {status === "error" ? (
                <div className="status-card status-card--error" role="alert">
                  <h2>Search unavailable</h2>
                  <p>{errorMessage}</p>
                </div>
              ) : null}

              {showEmptyState ? (
                <div className="status-card">
                  <h2>No results found</h2>
                  <p>
                    No SEP passages matched “{submittedQuery}”. Try a broader term, alternate
                    phrasing, or a related philosopher.
                  </p>
                </div>
              ) : null}

              {showResults ? (
                <>
                  <div className="results-panel__summary">
                    <p>
                      {results.length} passage{results.length === 1 ? "" : "s"} across {entryCount}{" "}
                      SEP {entryCount === 1 ? "entry" : "entries"} for “{submittedQuery}”
                    </p>
                    {topResult ? (
                      <button
                        className="results-panel__summary-action"
                        type="button"
                        onClick={() =>
                          handleOpenGraphForEntry({
                            slug: topResult.entry_slug,
                            title: topResult.entry_title,
                          })
                        }
                      >
                        Open top result in graph
                      </button>
                    ) : null}
                  </div>
                  <div className="results-list">
                    {groupedResults.map((group, index) => (
                      <EntryGroup
                        key={group.pageUrl}
                        group={group}
                        groupIndex={index}
                        onOpenGraph={handleOpenGraphForEntry}
                      />
                    ))}
                  </div>
                  <div className="results-panel__load-more">
                    {shouldShowLoadMore ? (
                      <button
                        className="results-panel__load-more-button"
                        type="button"
                        onClick={handleLoadMore}
                        disabled={isLoadingMore}
                      >
                        {isLoadingMore ? "Loading more..." : "Load more"}
                      </button>
                    ) : null}

                    {loadMoreErrorMessage ? (
                      <p className="results-panel__load-more-error" role="alert">
                        {loadMoreErrorMessage}
                      </p>
                    ) : null}
                  </div>
                </>
              ) : null}
              </section>
            ) : null}
          </>
        ) : (
          <>
            <section className="hero hero--graph">
              <h1>
                <button className="hero__brand" type="button" onClick={handleReturnHome}>
                  Zetesis
                </button>
              </h1>
              <p className="hero__copy">
                Trace how a selected SEP entry connects through direct references and strict
                semantic similarity.
              </p>
            </section>

            <GraphTab
              graphData={graphData}
              graphStatus={graphStatus}
              graphErrorMessage={graphErrorMessage}
              graphView={graphView}
              selectedGraphNode={selectedGraphNode}
              onBack={handleGraphBack}
              onBackToSearch={handleBackToSearchFromGraph}
              onGraphNodeSelect={handleOpenGraphForEntry}
              onSearchEntry={handleSearchEntryFromGraph}
              onRetryGraph={handleRetryGraph}
            />
          </>
        )}
      </main>
    </div>
  );
}
