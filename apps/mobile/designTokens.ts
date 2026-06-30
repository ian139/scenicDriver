export const scenicTokens = {
  color: {
    ink: '#17202A',
    muted: '#637080',
    faint: '#8D98A5',
    canvas: '#DFE6E3',
    paper: '#FCFDFC',
    wash: '#F4F7F5',
    line: 'rgba(31, 43, 55, 0.13)',
    lineStrong: 'rgba(31, 43, 55, 0.24)',
    brand: '#244F45',
    brandPressed: '#1D4038',
    brandSoft: 'rgba(36, 79, 69, 0.08)',
    brandRing: 'rgba(36, 79, 69, 0.12)',
    signal: '#B9653D',
    signalSoft: 'rgba(184, 100, 61, 0.08)',
    routeBlue: '#386F9F',
    routeBlueSoft: 'rgba(58, 111, 159, 0.08)',
    danger: '#8F3428',
    overlay: 'rgba(18, 27, 35, 0.38)',
  },
  space: {
    xxs: 2,
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    xxl: 32,
  },
  radius: {
    sm: 6,
    md: 8,
    lg: 12,
    pill: 999,
  },
  type: {
    family: 'Inter',
    micro: { fontSize: 10, lineHeight: 12, fontWeight: '800' as const, letterSpacing: 0.8 },
    label: { fontSize: 12, lineHeight: 16, fontWeight: '700' as const },
    body: { fontSize: 14, lineHeight: 20, fontWeight: '500' as const },
    title: { fontSize: 20, lineHeight: 22, fontWeight: '800' as const },
    metric: { fontSize: 18, lineHeight: 22, fontWeight: '800' as const },
  },
  shadow: {
    panel: {
      shadowColor: '#141E28',
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.11,
      shadowRadius: 28,
      elevation: 6,
    },
    card: {
      shadowColor: '#18212B',
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: 0.08,
      shadowRadius: 14,
      elevation: 3,
    },
  },
} as const;

export const scenicComponentMetrics = {
  inputHeight: 48,
  buttonHeight: 48,
  compactButtonHeight: 36,
  bottomSheetPeek: 104,
  routeCardMinHeight: 116,
  mapControlSize: 44,
} as const;

export const scenicRoutePresets = {
  fastest: { label: 'Fastest', scenicWeight: 0.15, maxDetourFactor: 1.25 },
  balanced: { label: 'Balanced', scenicWeight: 0.65, maxDetourFactor: 1.65 },
  scenic: { label: 'Scenic', scenicWeight: 0.9, maxDetourFactor: 2.4 },
} as const;

export const scenicStateCopy = {
  emptyRoute: 'Choose a start and destination to compare scenic and fastest routes.',
  loadingRoute: 'Comparing routes…',
  noRoute: 'No route found for this region. Try a closer start or destination.',
  apiError: 'Route request failed. Check the API connection and try again.',
  mapUnavailable: 'Map preview unavailable. Route comparison is still available.',
} as const;

export const scenicMobileBlueprint = {
  defaultRegion: 'masswhites',
  sheetStops: ['collapsed', 'planning', 'results'] as const,
  planningOrder: ['fromField', 'swapAction', 'toField', 'preferenceSegment', 'compareButton', 'statusBanner', 'advancedDisclosure'] as const,
  resultMetrics: ['time', 'distance', 'scenicScore'] as const,
  secondaryResultActions: ['share', 'clear', 'save'] as const,
  layersOrder: ['region', 'basemapStyle', 'scenicLayer', 'advancedApi'] as const,
} as const;
