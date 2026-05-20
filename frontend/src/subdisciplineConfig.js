import config from "../../shared/subdisciplines.json";

export const DEFAULT_SUBDISCIPLINE_COLOR = config.default_color;
export const SUBDISCIPLINE_COLORS = config.subdiscipline_colors;
export const SUBDISCIPLINE_LEGEND = config.legend_order;

export function getSubdisciplineColor(subdiscipline) {
  if (!subdiscipline) {
    return DEFAULT_SUBDISCIPLINE_COLOR;
  }

  return SUBDISCIPLINE_COLORS[subdiscipline] || DEFAULT_SUBDISCIPLINE_COLOR;
}
