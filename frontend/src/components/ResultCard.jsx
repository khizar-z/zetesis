import { useState } from "react";

const DISPLAY_LIMIT = 300;

function truncateText(text, limit = DISPLAY_LIMIT) {
  if (text.length <= limit) {
    return text;
  }

  const truncated = text.slice(0, limit);
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace <= limit * 0.6) {
    return `${truncated.trimEnd()}...`;
  }

  return `${truncated.slice(0, lastSpace).trimEnd()}...`;
}

export default function ResultCard({ result, index, compact = false }) {
  const [expanded, setExpanded] = useState(false);
  const canExpand = result.chunk_text.length > DISPLAY_LIMIT;
  const displayText = expanded ? result.chunk_text : truncateText(result.chunk_text);
  const sectionTitle = result.section_title?.trim() || "Entry overview";

  return (
    <article
      className={`result-card${compact ? " result-card--compact" : ""}`}
      style={{ animationDelay: `${index * 70}ms` }}
    >
      <header className="result-card__header">
        <div>
          <p className="result-card__eyebrow">{compact ? "Relevant Passage" : "SEP Passage"}</p>
          {compact ? <h3 className="result-card__section-heading">{sectionTitle}</h3> : <h2>{result.entry_title}</h2>}
          {compact ? null : <p className="result-card__section">{sectionTitle}</p>}
        </div>
        <a className="result-card__link" href={result.url} target="_blank" rel="noreferrer">
          {compact ? "Jump to section →" : "Read in SEP →"}
        </a>
      </header>

      <p className="result-card__text">{displayText}</p>

      {canExpand ? (
        <button
          className="result-card__toggle"
          type="button"
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </article>
  );
}
