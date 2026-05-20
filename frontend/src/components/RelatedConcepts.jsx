function formatSimilarity(similarity) {
  return `${Math.round(similarity * 100)}%`;
}

export default function RelatedConcepts({
  entryTitle,
  concepts,
  status,
  errorMessage,
  onOpenConcept,
}) {
  const showLoading = status === "loading";
  const showError = status === "error";
  const showConcepts = status === "success" && concepts.length > 0;

  if (!entryTitle && !showLoading && !showError && !showConcepts) {
    return null;
  }

  return (
    <section className="related-concepts">
      <div className="related-concepts__header">
        <div>
          <p className="related-concepts__eyebrow">Related Concepts</p>
          <h2>Follow the strongest semantic neighbors.</h2>
          <p className="related-concepts__copy">
            {entryTitle
              ? `These entries are the closest concept-graph matches to ${entryTitle}.`
              : "These entries are the strongest semantic neighbors of the top result."}
          </p>
        </div>
      </div>

      {showLoading ? (
        <div className="related-concepts__status" role="status" aria-live="polite">
          <div className="status-card__spinner" />
          <p>Finding nearby SEP concepts...</p>
        </div>
      ) : null}

      {showError ? (
        <div className="related-concepts__status related-concepts__status--error" role="alert">
          <p>{errorMessage}</p>
        </div>
      ) : null}

      {showConcepts ? (
        <div className="related-concepts__chips">
          {concepts.map((concept) => (
            <button
              key={concept.slug}
              type="button"
              className="related-concepts__chip"
              onClick={() => onOpenConcept(concept)}
            >
              <span>{concept.title}</span>
              <span className="related-concepts__chip-meta">
                {concept.subdiscipline || "Unclassified"} · {formatSimilarity(concept.similarity)}
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </section>
  );
}
