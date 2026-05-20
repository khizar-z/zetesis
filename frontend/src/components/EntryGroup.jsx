import ResultCard from "./ResultCard";

function pluralize(count, singular, plural) {
  return count === 1 ? singular : plural;
}

export default function EntryGroup({ group, groupIndex, onOpenGraph }) {
  const passageCount = group.results.length;

  return (
    <section className="entry-group" style={{ animationDelay: `${groupIndex * 90}ms` }}>
      <header className="entry-group__header">
        <div>
          <p className="entry-group__eyebrow">SEP Entry</p>
          <h2>{group.entryTitle}</h2>
          <p className="entry-group__meta">
            {passageCount} {pluralize(passageCount, "passage", "passages")} from this entry
          </p>
        </div>
        <div className="entry-group__actions">
          <button
            className="entry-group__button"
            type="button"
            onClick={() =>
              onOpenGraph({
                slug: group.entrySlug,
                title: group.entryTitle,
              })
            }
          >
            View neighborhood
          </button>
          <a className="entry-group__link" href={group.pageUrl} target="_blank" rel="noreferrer">
            Open entry →
          </a>
        </div>
      </header>

      <div className="entry-group__list">
        {group.results.map((result, passageIndex) => (
          <ResultCard
            key={`${result.url}-${passageIndex}`}
            result={result}
            index={group.startIndex + passageIndex}
            compact
          />
        ))}
      </div>
    </section>
  );
}
