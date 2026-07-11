import { useEffect, useMemo, useState } from 'react';

import { StatusBar } from 'expo-status-bar';
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  scenicComponentMetrics,
  scenicMobileBlueprint,
  scenicRoutePresets,
  scenicStateCopy,
  scenicTokens,
} from './designTokens';

type PreferenceKey = keyof typeof scenicRoutePresets;
type RouteKind = 'scenic' | 'baseline';
type StatusTone = 'info' | 'success' | 'error';

type LatLon = {
  lat: number;
  lon: number;
};

type Region = {
  region: string;
  display_name?: string;
  description?: string;
};

type GeocodeResult = LatLon & {
  label?: string;
  match_type?: string;
};

type RouteMetrics = {
  total_distance_km?: number;
  estimated_duration_minutes?: number;
  average_scenic_score?: number;
};

type RouteComparePayload = {
  run_name?: string;
  routes?: Partial<Record<RouteKind, RouteMetrics>>;
  deltas?: {
    distance_km?: number;
    duration_min?: number;
    scenic_score?: number;
  } | null;
  diagnostics?: {
    start_snap_km?: number;
    end_snap_km?: number;
  };
};

type RouteCardModel = {
  kind: RouteKind;
  title: string;
  badge: string;
  badgeStyle: 'warm' | 'cool';
  values: Record<'time' | 'distance' | 'scenicScore', string>;
  deltas: string[];
};

const DEFAULT_REGIONS: Region[] = [
  { region: 'masswhites', display_name: 'Masswhites' },
  { region: 'philadelphia', display_name: 'Philadelphia' },
  { region: 'pittsfield', display_name: 'Pittsfield' },
];

const metricLabels = {
  time: 'Time',
  distance: 'Distance',
  scenicScore: 'Scenic',
} as const;

function apiUrl(baseUrl: string, path: string, params?: Record<string, string>) {
  const root = baseUrl.trim().replace(/\/+$/, '');
  const query = params
    ? `?${Object.entries(params)
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
        .join('&')}`
    : '';
  return `${root}${path}${query}`;
}

function parseCoordinateInput(value: string): LatLon | null {
  const match = value.trim().match(/^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$/);
  if (!match) return null;
  const lat = Number(match[1]);
  const lon = Number(match[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return { lat, lon };
}

function formatDistance(km?: number) {
  const miles = Number(km) * 0.621371;
  return Number.isFinite(miles) ? `${miles.toFixed(1)} mi` : 'n/a';
}

function formatMinutes(minutes?: number) {
  const value = Number(minutes);
  return Number.isFinite(value) ? `${value.toFixed(1)} min` : 'n/a';
}

function formatScore(score?: number) {
  const value = Number(score);
  return Number.isFinite(value) ? value.toFixed(2) : 'n/a';
}

function formatDelta(value: number | undefined, suffix: string) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return `±0.0${suffix}`;
  const sign = numeric > 0 ? '+' : numeric < 0 ? '-' : '±';
  return `${sign}${Math.abs(numeric).toFixed(1)}${suffix}`;
}


function routeCardsFromPayload(payload: RouteComparePayload | null): RouteCardModel[] {
  if (!payload?.routes) return [];
  const cards: RouteCardModel[] = [];
  const scenic = payload.routes.scenic;
  const baseline = payload.routes.baseline;
  const deltas = payload.deltas || undefined;

  if (scenic) {
    cards.push({
      kind: 'scenic',
      title: 'Scenic',
      badge: deltas?.duration_min ? formatDelta(deltas.duration_min, ' min') : 'recommended',
      badgeStyle: 'warm',
      values: {
        time: formatMinutes(scenic.estimated_duration_minutes),
        distance: formatDistance(scenic.total_distance_km),
        scenicScore: formatScore(scenic.average_scenic_score),
      },
      deltas: [formatDelta(deltas?.duration_min, ' min'), formatDelta(deltas?.scenic_score, ' scenic')],
    });
  }

  if (baseline) {
    cards.push({
      kind: 'baseline',
      title: 'Fastest',
      badge: 'baseline',
      badgeStyle: 'cool',
      values: {
        time: formatMinutes(baseline.estimated_duration_minutes),
        distance: formatDistance(baseline.total_distance_km),
        scenicScore: formatScore(baseline.average_scenic_score),
      },
      deltas: [],
    });
  }

  return cards;
}

async function readErrorMessage(response: Response) {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object' && 'message' in detail) {
      const message = (detail as { message?: unknown }).message;
      const hint = (detail as { hint?: unknown }).hint;
      return [message, hint].filter((value): value is string => typeof value === 'string').join(' ');
    }
    return JSON.stringify(detail);
  }

  return `Request failed (${response.status}).`;
}

function Field({
  label,
  value,
  onChangeText,
  placeholder,
}: {
  label: string;
  value: string;
  onChangeText: (value: string) => void;
  placeholder: string;
}) {
  return (
    <View style={styles.fieldGroup}>
      <Text style={styles.microLabel}>{label}</Text>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor={scenicTokens.color.faint}
        accessibilityLabel={`${label} address or coordinates`}
        autoCorrect={false}
        style={styles.input}
      />
    </View>
  );
}

function PreferenceSegment({ value, onChange }: { value: PreferenceKey; onChange: (value: PreferenceKey) => void }) {
  return (
    <View accessibilityLabel="Route preference" style={styles.segment}>
      {Object.entries(scenicRoutePresets).map(([key, preset]) => {
        const presetKey = key as PreferenceKey;
        const active = presetKey === value;
        return (
          <Pressable
            key={key}
            accessibilityRole="button"
            accessibilityState={{ selected: active }}
            onPress={() => onChange(presetKey)}
            style={[styles.segmentButton, active && styles.segmentButtonActive]}
          >
            <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{preset.label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function RouteCard({ card, active, onPress }: { card: RouteCardModel; active: boolean; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: active }}
      accessibilityLabel={`${card.title} route option`}
      onPress={onPress}
      style={[styles.routeCard, active && styles.routeCardActive]}
    >
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>{card.title}</Text>
        <Text style={[styles.badge, card.badgeStyle === 'warm' ? styles.badgeWarm : styles.badgeCool]}>{card.badge}</Text>
      </View>
      <View style={styles.metricRow}>
        {scenicMobileBlueprint.resultMetrics.map((metric) => (
          <View key={metric} style={styles.metricBlock}>
            <Text style={styles.microLabel}>{metricLabels[metric]}</Text>
            <Text style={styles.metricValue}>{card.values[metric]}</Text>
          </View>
        ))}
      </View>
      {card.deltas.length > 0 ? (
        <View style={styles.deltaRow}>
          {card.deltas.map((delta) => (
            <Text key={delta} style={styles.deltaChip}>
              {delta}
            </Text>
          ))}
        </View>
      ) : null}
    </Pressable>
  );
}

function StatusBanner({ tone, message }: { tone: StatusTone; message: string }) {
  return (
    <View style={[styles.statusBanner, tone === 'error' && styles.statusError, tone === 'success' && styles.statusSuccess]}>
      <Text style={[styles.statusText, tone === 'error' && styles.statusErrorText]}>{message}</Text>
    </View>
  );
}

export default function App() {
  const [from, setFrom] = useState('Pittsfield, Massachusetts');
  const [to, setTo] = useState('Great Barrington, Massachusetts');
  const [preference, setPreference] = useState<PreferenceKey>('balanced');
  const [selectedRoute, setSelectedRoute] = useState<RouteKind>('scenic');
  const [apiBase, setApiBase] = useState('http://localhost:8080');
  const [regions, setRegions] = useState<Region[]>(DEFAULT_REGIONS);
  const [region, setRegion] = useState(scenicMobileBlueprint.defaultRegion);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ tone: StatusTone; message: string }>({
    tone: 'info',
    message: scenicStateCopy.emptyRoute,
  });
  const [routePayload, setRoutePayload] = useState<RouteComparePayload | null>(null);
  const routeCards = useMemo(() => routeCardsFromPayload(routePayload), [routePayload]);
  const selectedPreset = scenicRoutePresets[preference];

  useEffect(() => {
    let cancelled = false;
    async function loadRegions() {
      try {
        const response = await fetch(apiUrl(apiBase, '/v1/regions'));
        if (!response.ok) return;
        const payload = (await response.json()) as { regions?: Region[] };
        if (!cancelled && Array.isArray(payload.regions) && payload.regions.length > 0) {
          setRegions(payload.regions);
          if (!payload.regions.some((item) => item.region === region)) {
            setRegion(payload.regions[0].region);
          }
        }
      } catch {
        // Keep the local MVP region list; route planning will surface connection errors.
      }
    }
    loadRegions();
    return () => {
      cancelled = true;
    };
  }, [apiBase, region]);

  async function resolvePoint(query: string): Promise<GeocodeResult> {
    const coordinate = parseCoordinateInput(query);
    if (coordinate) return { ...coordinate, label: `${coordinate.lat}, ${coordinate.lon}`, match_type: 'coordinate' };

    const response = await fetch(apiUrl(apiBase, '/v1/geocode', { q: query, region }));
    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }
    const payload = (await response.json()) as { results?: GeocodeResult[] };
    const [first] = payload.results || [];
    if (!first) throw new Error(`No geocoding result for “${query}”. Try coordinates like 42.45, -73.25.`);
    return first;
  }

  async function compareRoute() {
    if (loading) return;
    setLoading(true);
    setStatus({ tone: 'info', message: scenicStateCopy.loadingRoute });
    try {
      const [start, end] = await Promise.all([resolvePoint(from), resolvePoint(to)]);
      const response = await fetch(apiUrl(apiBase, '/v1/route/compare'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          start: { lat: start.lat, lon: start.lon },
          end: { lat: end.lat, lon: end.lon },
          scenic_weight: selectedPreset.scenicWeight,
          max_detour_factor: selectedPreset.maxDetourFactor,
          region,
          avoid_highways: false,
          include_baseline: true,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      const payload = (await response.json()) as RouteComparePayload;
      setRoutePayload(payload);
      setSelectedRoute(payload.routes?.scenic ? 'scenic' : 'baseline');
      const diagnostics = payload.diagnostics;
      const snapMessage = diagnostics?.start_snap_km || diagnostics?.end_snap_km
        ? ` Snapped to road graph: start ${Number(diagnostics.start_snap_km || 0).toFixed(2)} km, end ${Number(
            diagnostics.end_snap_km || 0,
          ).toFixed(2)} km.`
        : '';
      setStatus({
        tone: 'success',
        message: `Compared ${start.label || 'start'} to ${end.label || 'destination'}.${snapMessage}`,
      });
    } catch (error) {
      setStatus({ tone: 'error', message: error instanceof Error ? error.message : scenicStateCopy.apiError });
    } finally {
      setLoading(false);
    }
  }

  function clearRoute() {
    setRoutePayload(null);
    setSelectedRoute('scenic');
    setStatus({ tone: 'info', message: scenicStateCopy.emptyRoute });
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <View style={styles.mapPreview}>
        <View style={styles.mapHeader}>
          <View>
            <Text style={styles.brandTitle}>ScenicDriver</Text>
            <Text style={styles.mapEyebrow}>{regions.find((item) => item.region === region)?.display_name || region}</Text>
          </View>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Open layers and settings"
            onPress={() => setAdvancedOpen((open) => !open)}
            style={styles.layersButton}
          >
            <Text style={styles.layersText}>{advancedOpen ? 'Close' : 'Layers'}</Text>
          </Pressable>
        </View>
        <View style={styles.mapPlaceholder}>
          <View style={styles.surfaceGrid}>
            {Array.from({ length: 24 }).map((_, index) => (
              <View
                key={index}
                style={[
                  styles.surfaceCell,
                  index % 5 === 0 && styles.surfaceCellPeak,
                  index % 7 === 0 && styles.surfaceCellWarm,
                ]}
              />
            ))}
          </View>
          <View style={[styles.routeRibbon, selectedRoute === 'baseline' && styles.routeRibbonBaseline]} />
          <Text style={styles.mapPlaceholderTitle}>{selectedRoute === 'baseline' ? 'Fastest route' : 'Scenic route'}</Text>
          <Text style={styles.mapPlaceholderBody}>{scenicStateCopy.mapUnavailable}</Text>
        </View>
      </View>

      <View style={styles.sheet}>
        <View style={styles.sheetGrip} />
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.sheetContent} keyboardShouldPersistTaps="handled">
          <View style={styles.routeSummary}>
            <Text style={styles.microLabel}>{routePayload?.run_name || region}</Text>
            <Text style={styles.summaryTitle}>Compare a better drive</Text>
            <Text style={styles.summaryBody}>Keep the fastest route honest against the scenic tradeoff.</Text>
          </View>

          <Field label="From" value={from} onChangeText={setFrom} placeholder="Address or 42.45, -73.25" />
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Swap start and destination"
            onPress={() => {
              setFrom(to);
              setTo(from);
            }}
            style={styles.swapButton}
          >
            <Text style={styles.swapText}>Swap</Text>
          </Pressable>
          <Field label="To" value={to} onChangeText={setTo} placeholder="Address or 42.50, -73.20" />
          <PreferenceSegment value={preference} onChange={setPreference} />

          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Compare route"
            accessibilityState={{ disabled: loading }}
            disabled={loading}
            onPress={compareRoute}
            style={[styles.primaryButton, loading && styles.primaryButtonDisabled]}
          >
            {loading ? <ActivityIndicator color={scenicTokens.color.paper} /> : <Text style={styles.primaryButtonText}>Compare Route</Text>}
          </Pressable>

          <StatusBanner tone={status.tone} message={status.message} />

          {advancedOpen ? (
            <View style={styles.advancedPanel}>
              <View style={styles.resultsHeader}>
                <View>
                  <Text style={styles.microLabel}>Layers</Text>
                  <Text style={styles.sectionTitle}>Route settings</Text>
                </View>
                <Text style={styles.advancedText}>{selectedPreset.scenicWeight.toFixed(2)} scenic</Text>
              </View>
              <View style={styles.regionList}>
                {regions.map((item) => {
                  const active = item.region === region;
                  return (
                    <Pressable
                      key={item.region}
                      accessibilityRole="button"
                      accessibilityState={{ selected: active }}
                      onPress={() => setRegion(item.region)}
                      style={[styles.regionChip, active && styles.regionChipActive]}
                    >
                      <Text style={[styles.regionChipText, active && styles.regionChipTextActive]}>{item.display_name || item.region}</Text>
                    </Pressable>
                  );
                })}
              </View>
              <View style={styles.fieldGroup}>
                <Text style={styles.microLabel}>API Base URL</Text>
                <TextInput
                  value={apiBase}
                  onChangeText={setApiBase}
                  accessibilityLabel="API Base URL"
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                  style={styles.input}
                />
              </View>
              <Text style={styles.knobCopy}>
                {selectedPreset.label}: weight {selectedPreset.scenicWeight.toFixed(2)}, detour cap{' '}
                {selectedPreset.maxDetourFactor.toFixed(2)}x
              </Text>
            </View>
          ) : null}

          <View style={styles.resultsHeader}>
            <View>
              <Text style={styles.microLabel}>Compare</Text>
              <Text style={styles.sectionTitle}>Scenic vs fastest</Text>
            </View>
            {routePayload ? (
              <Pressable accessibilityRole="button" accessibilityLabel="Clear route results" onPress={clearRoute} style={styles.clearButton}>
                <Text style={styles.clearButtonText}>Clear</Text>
              </Pressable>
            ) : null}
          </View>

          {routeCards.length > 0 ? (
            routeCards.map((card) => (
              <RouteCard key={card.kind} card={card} active={selectedRoute === card.kind} onPress={() => setSelectedRoute(card.kind)} />
            ))
          ) : (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyTitle}>No route yet</Text>
              <Text style={styles.emptyBody}>{scenicStateCopy.emptyRoute}</Text>
            </View>
          )}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: scenicTokens.color.canvas,
  },
  mapPreview: {
    flex: 1,
    padding: scenicTokens.space.lg,
  },
  mapHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  brandTitle: {
    ...scenicTokens.type.title,
    color: scenicTokens.color.ink,
  },
  mapEyebrow: {
    ...scenicTokens.type.micro,
    color: scenicTokens.color.muted,
    marginTop: scenicTokens.space.xs,
    textTransform: 'uppercase',
  },
  layersButton: {
    minHeight: scenicComponentMetrics.mapControlSize,
    paddingHorizontal: scenicTokens.space.md,
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.paper,
    justifyContent: 'center',
    ...scenicTokens.shadow.card,
  },
  layersText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.ink,
  },
  mapPlaceholder: {
    flex: 1,
    marginTop: scenicTokens.space.lg,
    borderRadius: scenicTokens.radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.wash,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    padding: scenicTokens.space.xl,
  },
  surfaceGrid: {
    ...StyleSheet.absoluteFillObject,
    flexDirection: 'row',
    flexWrap: 'wrap',
    opacity: 0.62,
  },
  surfaceCell: {
    width: '16.666%',
    height: '25%',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.42)',
    backgroundColor: 'rgba(64, 125, 105, 0.22)',
  },
  surfaceCellWarm: {
    backgroundColor: 'rgba(230, 176, 86, 0.32)',
  },
  surfaceCellPeak: {
    backgroundColor: 'rgba(151, 68, 52, 0.28)',
  },
  routeRibbon: {
    position: 'absolute',
    width: '78%',
    height: 10,
    borderRadius: scenicTokens.radius.pill,
    backgroundColor: scenicTokens.color.signal,
    transform: [{ rotate: '-18deg' }],
    opacity: 0.9,
  },
  routeRibbonBaseline: {
    backgroundColor: scenicTokens.color.routeBlue,
    transform: [{ rotate: '8deg' }],
  },
  mapPlaceholderTitle: {
    ...scenicTokens.type.metric,
    color: scenicTokens.color.brand,
    marginTop: scenicTokens.space.xxl,
  },
  mapPlaceholderBody: {
    ...scenicTokens.type.body,
    color: scenicTokens.color.muted,
    marginTop: scenicTokens.space.sm,
    textAlign: 'center',
  },
  sheet: {
    maxHeight: '72%',
    borderTopLeftRadius: scenicTokens.radius.lg,
    borderTopRightRadius: scenicTokens.radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.paper,
    paddingTop: scenicTokens.space.sm,
    ...scenicTokens.shadow.panel,
  },
  sheetGrip: {
    width: 36,
    height: scenicTokens.space.xs,
    borderRadius: scenicTokens.radius.pill,
    backgroundColor: scenicTokens.color.lineStrong,
    alignSelf: 'center',
    marginBottom: scenicTokens.space.sm,
  },
  sheetContent: {
    paddingHorizontal: scenicTokens.space.lg,
    paddingBottom: scenicTokens.space.xxl,
    rowGap: scenicTokens.space.md,
  },
  routeSummary: {
    rowGap: scenicTokens.space.xs,
  },
  summaryTitle: {
    ...scenicTokens.type.title,
    color: scenicTokens.color.ink,
  },
  summaryBody: {
    ...scenicTokens.type.body,
    color: scenicTokens.color.muted,
  },
  fieldGroup: {
    rowGap: scenicTokens.space.sm,
  },
  microLabel: {
    ...scenicTokens.type.micro,
    color: scenicTokens.color.faint,
    textTransform: 'uppercase',
  },
  input: {
    minHeight: scenicComponentMetrics.inputHeight,
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.paper,
    paddingHorizontal: scenicTokens.space.md,
    ...scenicTokens.type.body,
    color: scenicTokens.color.ink,
  },
  swapButton: {
    alignSelf: 'flex-end',
    minHeight: scenicComponentMetrics.compactButtonHeight,
    justifyContent: 'center',
    paddingHorizontal: scenicTokens.space.md,
    borderRadius: scenicTokens.radius.md,
    backgroundColor: scenicTokens.color.wash,
  },
  swapText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.muted,
  },
  segment: {
    flexDirection: 'row',
    padding: scenicTokens.space.xxs,
    borderRadius: scenicTokens.radius.md,
    backgroundColor: scenicTokens.color.wash,
  },
  segmentButton: {
    flex: 1,
    minHeight: scenicComponentMetrics.compactButtonHeight,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: scenicTokens.radius.sm,
  },
  segmentButtonActive: {
    backgroundColor: scenicTokens.color.paper,
    ...scenicTokens.shadow.card,
  },
  segmentText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.muted,
  },
  segmentTextActive: {
    color: scenicTokens.color.ink,
  },
  primaryButton: {
    minHeight: scenicComponentMetrics.buttonHeight,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: scenicTokens.radius.md,
    backgroundColor: scenicTokens.color.brand,
  },
  primaryButtonDisabled: {
    opacity: 0.68,
  },
  primaryButtonText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.paper,
  },
  statusBanner: {
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.brandSoft,
    padding: scenicTokens.space.md,
  },
  statusSuccess: {
    borderColor: scenicTokens.color.brand,
  },
  statusError: {
    borderColor: scenicTokens.color.danger,
    backgroundColor: 'rgba(143, 52, 40, 0.08)',
  },
  statusText: {
    ...scenicTokens.type.body,
    color: scenicTokens.color.muted,
  },
  statusErrorText: {
    color: scenicTokens.color.danger,
  },
  advancedPanel: {
    borderRadius: scenicTokens.radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.wash,
    padding: scenicTokens.space.md,
    rowGap: scenicTokens.space.md,
  },
  regionList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: scenicTokens.space.sm,
  },
  regionChip: {
    minHeight: scenicComponentMetrics.compactButtonHeight,
    justifyContent: 'center',
    borderRadius: scenicTokens.radius.pill,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.paper,
    paddingHorizontal: scenicTokens.space.md,
  },
  regionChipActive: {
    borderColor: scenicTokens.color.brand,
    backgroundColor: scenicTokens.color.brandSoft,
  },
  regionChipText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.muted,
  },
  regionChipTextActive: {
    color: scenicTokens.color.brand,
  },
  knobCopy: {
    ...scenicTokens.type.body,
    color: scenicTokens.color.muted,
  },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    marginTop: scenicTokens.space.sm,
    gap: scenicTokens.space.md,
  },
  sectionTitle: {
    ...scenicTokens.type.metric,
    color: scenicTokens.color.ink,
  },
  advancedText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.muted,
  },
  clearButton: {
    minHeight: scenicComponentMetrics.compactButtonHeight,
    justifyContent: 'center',
    borderRadius: scenicTokens.radius.md,
    backgroundColor: scenicTokens.color.wash,
    paddingHorizontal: scenicTokens.space.md,
  },
  clearButtonText: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.muted,
  },
  routeCard: {
    minHeight: scenicComponentMetrics.routeCardMinHeight,
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.paper,
    padding: scenicTokens.space.md,
    rowGap: scenicTokens.space.md,
  },
  routeCardActive: {
    borderColor: scenicTokens.color.brand,
    ...scenicTokens.shadow.card,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: scenicTokens.space.md,
  },
  cardTitle: {
    ...scenicTokens.type.body,
    fontWeight: '800',
    color: scenicTokens.color.ink,
  },
  badge: {
    ...scenicTokens.type.micro,
    overflow: 'hidden',
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: scenicTokens.space.sm,
    paddingVertical: scenicTokens.space.xs,
    textTransform: 'uppercase',
  },
  badgeWarm: {
    color: scenicTokens.color.signal,
    borderColor: scenicTokens.color.signal,
    backgroundColor: scenicTokens.color.signalSoft,
  },
  badgeCool: {
    color: scenicTokens.color.routeBlue,
    borderColor: scenicTokens.color.routeBlue,
    backgroundColor: scenicTokens.color.routeBlueSoft,
  },
  metricRow: {
    flexDirection: 'row',
    columnGap: scenicTokens.space.sm,
  },
  metricBlock: {
    flex: 1,
    rowGap: scenicTokens.space.xs,
  },
  metricValue: {
    ...scenicTokens.type.metric,
    color: scenicTokens.color.ink,
  },
  deltaRow: {
    flexDirection: 'row',
    columnGap: scenicTokens.space.sm,
  },
  deltaChip: {
    ...scenicTokens.type.label,
    color: scenicTokens.color.muted,
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.wash,
    paddingHorizontal: scenicTokens.space.sm,
    paddingVertical: scenicTokens.space.xs,
  },
  emptyCard: {
    borderRadius: scenicTokens.radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: scenicTokens.color.line,
    backgroundColor: scenicTokens.color.wash,
    padding: scenicTokens.space.lg,
    rowGap: scenicTokens.space.sm,
  },
  emptyTitle: {
    ...scenicTokens.type.body,
    fontWeight: '800',
    color: scenicTokens.color.ink,
  },
  emptyBody: {
    ...scenicTokens.type.body,
    color: scenicTokens.color.muted,
  },
});
