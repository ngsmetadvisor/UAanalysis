# -*- coding: utf-8 -*-

# ── Standard library ──────────────────────────────────────────
import csv, io, json, math, re, time, warnings
import concurrent.futures
from collections import defaultdict

# ── Third-party: core ─────────────────────────────────────────
import numpy as np
import pandas as pd
import requests

# ── Third-party: matplotlib (backend before pyplot) ───────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Third-party: scipy ────────────────────────────────────────
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (gaussian_filter, maximum_filter,
                           minimum_filter, label)

# ── Third-party: geo / mapping ────────────────────────────────
import folium
import branca
from shapely.geometry import shape

# ── Third-party: kriging ──────────────────────────────────────
from pykrige.ok import OrdinaryKriging

print('✓ All packages ready')

# -- Cell 1.5 - Configuration ------------------------------------------



CSV_URL         = 'https://raw.githubusercontent.com/ngsmetadvisor/SfcMap/main/AP_location.csv'
METAR_API       = 'https://aviationweather.gov/api/data/metar'
COVERAGE        = 'standard'       # essential | standard | all | chart
EXPORT_TIME     = '1200Z'          # 0000Z | 0600Z | 1200Z | 1800Z
INTERP_METHOD   = 'rbf'            # rbf | kriging
SLP_INTERVAL    = 4                # hPa spacing between isobars (standard=4)
GRID_N          = 240              # grid points per axis (60–600, step 20)
RBF_SMOOTHING   = 0.0              # 0.0 = exact fit; 0.2–0.5 typical for SLP
SIGMA_SMOOTH    = 1.0              # gaussian blur after interpolation (2–4 typical)
SYMBOL_SCALE    = 28               # station model size px (10–80)
FONT_SCALE      = 10               # station label font size (4–20)
HL_NEIGHBORHOOD = 5                # H/L search radius in grid cells (1–60)
HL_MIN_DELTA    = 0.5              # min pressure diff hPa to accept a centre (0.1–10.0)
HL_SIGMA        = 1.0              # gaussian smooth before extrema search (0.1–20.0)

print(f'Coverage: {COVERAGE} | Interp: {INTERP_METHOD} | Grid: {GRID_N} | Export: {EXPORT_TIME}')


# ====================================================
#=====================================================
import threading, time

##########################################################




#####################################################










# ====================================================
#=====================================================
import threading, time

##########################################################

import urllib.request, xml.etree.ElementTree as ET, json

_KML_URL = 'http://orangecore.net/met/wxchart/Alberta_Fire_Weather_Forecast_Zones.kml'

def _kml_to_geojson(kml_bytes):
    root = ET.fromstring(kml_bytes)
    features = []
    for pm in root.iter('Placemark'):
        name = ''
        for sd in pm.iter('SimpleData'):
            if sd.get('name') == 'NAME':
                name = sd.text or ''
                break
        if not name:
            n_tag = pm.find('name')
            name  = n_tag.text if n_tag is not None else 'Unknown'

        wfz_id = ''
        for sd in pm.iter('SimpleData'):
            if sd.get('name') == 'WFZ_ID':
                wfz_id = sd.text or ''
                break

        rings = []
        for coords_el in pm.findall('.//coordinates'):
            pts = []
            for token in coords_el.text.strip().split():
                parts = token.split(',')
                if len(parts) >= 2:
                    pts.append([float(parts[0]), float(parts[1])])
            if pts:
                rings.append(pts)

        if not rings:
            continue

        features.append({
            'type': 'Feature',
            'properties': {'name': name, 'wfz_id': wfz_id},
            'geometry':   {'type': 'Polygon', 'coordinates': rings}
        })

    return {'type': 'FeatureCollection', 'features': features}

# ── Fetch ──────────────────────────────────────────────────────────────────
try:
    with urllib.request.urlopen(_KML_URL, timeout=15) as _r:
        _kml_bytes = _r.read()
    _fire_zones_geojson_str = json.dumps(_kml_to_geojson(_kml_bytes))
    _zone_count = len(json.loads(_fire_zones_geojson_str)['features'])
    print(f'Alberta Fire Zone KML fetched → {_zone_count} zones loaded')
except Exception as e:
    print(f'WARNING: Fire zone KML fetch failed ({e}) — layer will be skipped')
    _fire_zones_geojson_str = '{"type":"FeatureCollection","features":[]}'

# ── Build HTML ─────────────────────────────────────────────────────────────
fire_zones_html = (
    '<script>\n'
    'var _FIRE_ZONES_GEOJSON = ' + _fire_zones_geojson_str + ';\n'
    '(function() {\n'
    '  function loadFireZones() {\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if (!keys.length) { setTimeout(loadFireZones, 300); return; }\n'
    '    var MAP = window[keys[0]];\n'
    '    var fireLayer = L.geoJSON(_FIRE_ZONES_GEOJSON, {\n'
    '      style: function() {\n'
    '        return {\n'
    '          color: "#cc0000",\n'
    '          weight: 1.8,\n'
    '          opacity: 0.85,\n'
    '          fillColor: "#ff9933",\n'
    '          fillOpacity: 0.00,\n'
    '          dashArray: "4 3"\n'
    '        };\n'
    '      },\n'
    '      onEachFeature: function(feature, layer) {\n'
    '        var name = (feature.properties && feature.properties.name) || "Fire Zone";\n'
    '        layer.bindTooltip(name, {sticky: true, opacity: 0.9});\n'
    '        layer.bindPopup(\n'
    '          \'<div style="font-family:Courier New,monospace;font-size:12px;">\'\n'
    '          + \'<b style="color:#cc4400">\' + name + \'</b><br>\'\n'
    '          + \'Alberta Fire Weather Forecast Zone\'\n'
    '          + \'</div>\'\n'
    '        );\n'
    '      }\n'
    '    });\n'
    '    var _fireVisible = true;\n'
    '    var btn = document.getElementById("btn-fire-zones");\n'
    '    if (btn) {\n'
    '      btn.onclick = function() {\n'
    '        _fireVisible = !_fireVisible;\n'
    '        if (_fireVisible) { fireLayer.addTo(MAP); btn.style.background = "#cc4400"; }\n'
    '        else { MAP.removeLayer(fireLayer); btn.style.background = "#b0b8c8"; }\n'
    '      };\n'
    '      btn.style.background = "#cc4400";\n'
    '    }\n'
    '    fireLayer.addTo(MAP);\n'
    '  }\n'
    '  if (document.readyState === "complete") { setTimeout(loadFireZones, 800); }\n'
    '  else { window.addEventListener("load", function(){ setTimeout(loadFireZones, 800); }); }\n'
    '})();\n'
    '</script>\n'
    '<style>#btn-fire-zones { transition: background 0.2s; }</style>\n'
)

print('fire_zones_html ready')

# ══════════════════════════════════════════════════════════════════════════
#  STATION MODEL CONTROLS  ← edit these values only
# ══════════════════════════════════════════════════════════════════════════

# ── Circle ────────────────────────────────────────────────────────────────
CIRCLE_RADIUS   = 0.14   # fraction of S  (0.10 small → 0.20 large)

# ── Wind barb ─────────────────────────────────────────────────────────────
BARB_STAFF_LEN  = 1.20   # staff length multiplier of S
BARB_FULL_LEN   = 0.40   # full-barb (10kt) length fraction of S
BARB_HALF_LEN   = 0.20   # half-barb (5kt) length fraction of S
BARB_SPACING    = 0.12 # spacing between barbs fraction of S
BARB_LINE_WIDTH = 0.038  # stroke width fraction of S  (min 0.9px)
FEATHER_ANGLE   = 110    # degrees from staff  (90=perp, >90 tilts toward tip)
FEATHER_SIDE    = +1     # +1=right side, -1=left side (WMO standard)

# ── Font ──────────────────────────────────────────────────────────────────
FONT_SIZE_SCALE = 0.6   # font size = max(FONT_MIN_PX, S * this)
FONT_MIN_PX     = 12     # absolute minimum font size in px

# ── Label spacing ─────────────────────────────────────────────────────────
LABEL_HORIZ_OFF = 0.18   # horizontal offset from circle edge, fraction of S
LABEL_VERT_OFF  = 20     # vertical offset top/bottom labels from centre (px)
LABEL_ROW_GAP   = 1.2    # multiplier between name/ceiling rows

# ── Canvas ────────────────────────────────────────────────────────────────
CANVAS_PAD      = 1.5    # padding fraction of S
CANVAS_H_FACTOR = 3.4    # canvas height = S * this + PAD*2

# ══════════════════════════════════════════════════════════════════════════
print('Open this session to change the met symbols sizes')

# -- Cell 8 - WMO station model as SVG string ---
import math




def cloud_circle_svg(cx, cy, R, oktas):
    lw = max(0.9, R * 0.13)
    s = []
    if oktas == 9:  # VV — full black + white X
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<line x1="{cx-R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx+R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        s.append(f'<line x1="{cx+R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx-R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        return ''.join(s)
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="white" stroke="black" stroke-width="{lw}"/>')
    if oktas <= 0:
        return ''.join(s)
    if oktas >= 8:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        return ''.join(s)
    if oktas == 2:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 0,1 {cx+R:.2f},{cy} Z" fill="black"/>')
    elif oktas == 4:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 1,1 {cx},{cy+R:.2f} Z" fill="black"/>')
    elif oktas == 6:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<path d="M{cx},{cy} L{cx-R:.2f},{cy} A{R:.2f},{R:.2f} 0 0,1 {cx},{cy-R:.2f} Z" fill="white"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="black" stroke-width="{lw}"/>')
    return ''.join(s)


def wind_barb_svg(cx, cy, R, wind_dir, wind_spd, wind_gust, S):
    """WMO wind barb — direction/speed controlled by module constants."""
    if wind_dir is None or wind_spd is None:
        return ''
    if wind_spd < 3:
        return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R*1.5:.2f}" '
                f'fill="none" stroke="black" stroke-width="1"/>')

    sl   = S * BARB_STAFF_LEN
    blen = S * BARB_FULL_LEN
    bspc = S * BARB_SPACING
    lw   = max(0.9, S * BARB_LINE_WIDTH)

    staff_base_y = -R
    staff_tip_y  = -(R + sl)

    fx_full = FEATHER_SIDE * blen
    fx_half = FEATHER_SIDE * S * BARB_HALF_LEN
    tilt    = math.tan(math.radians(FEATHER_ANGLE - 90)) * blen

    spd = int(round(wind_spd / 5.0)) * 5
    pn  = spd // 50;  spd -= pn * 50
    fu  = spd // 10;  spd -= fu * 10
    ha  = spd //  5

    parts = []
    parts.append(
        f'<line x1="0" y1="{staff_base_y:.2f}" x2="0" y2="{staff_tip_y:.2f}" '
        f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
    )

    pos = 0.0

    if pn == 0 and fu == 0 and ha == 1:
        # lone half-barb — draw slightly inset from tip
        hy = staff_tip_y + 0.28 * sl
        parts.append(
            f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
            f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
        )
    else:
        for _ in range(pn):  # 50-kt pennants
            ay  = staff_tip_y + pos
            by2 = staff_tip_y + pos + bspc * 2
            pts = f'0,{ay:.2f} {fx_full:.2f},{ay - tilt:.2f} 0,{by2:.2f}'
            parts.append(f'<polygon points="{pts}" fill="black"/>')
            pos += bspc * 1.5
        for _ in range(fu):  # 10-kt full barbs
            fy = staff_tip_y + pos
            parts.append(
                f'<line x1="0" y1="{fy:.2f}" x2="{fx_full:.2f}" y2="{fy - tilt:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
            )
            pos += bspc
        for _ in range(ha):  # 5-kt half barbs
            hy = staff_tip_y + pos
            parts.append(
                f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>'
            )
            pos += bspc

    inner = ''.join(parts)
    return (
        f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({wind_dir:.1f})">'
        f'{inner}</g>'
    )


def pressure_tendency_svg(cx, cy, R, tendency, S, fs):
    """
    WMO pressure tendency symbol to the right of the station circle.
    Accepts int codes (0–7) or string keys.
    """
    _map = {
        'rising':         2,
        'falling':        7,
        'steady':         4,
        'rising_falling': 0,
        'falling_rising': 5,
        'rising_steady':  1,
        'falling_steady': 6,
    }
    if isinstance(tendency, str):
        tendency = _map.get(tendency.lower())
    if tendency is None:
        return ''

    lw   = max(0.9, S * 0.042)
    off  = R + S * LABEL_HORIZ_OFF

    # align with the pressure-change number row
    ox   = cx + off + fs * 2.2
    oy   = cy - R * 0.6 - LABEL_VERT_OFF + S * 0.65

    arm  = S * 0.22
    rise = S * 0.20

    def seg(x1, y1, x2, y2):
        return (
            f'<line x1="{ox+x1:.2f}" y1="{oy+y1:.2f}" '
            f'x2="{ox+x2:.2f}" y2="{oy+y2:.2f}" '
            f'stroke="black" stroke-width="{lw:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    parts = []
    if   tendency == 2: parts.append(seg(-arm,  rise*.5,  arm, -rise*.5))
    elif tendency == 7: parts.append(seg(-arm, -rise*.5,  arm,  rise*.5))
    elif tendency == 4: parts.append(seg(-arm,  0,        arm,  0))
    elif tendency == 0:
        parts += [seg(-arm, rise*.5, 0, -rise*.5), seg(0, -rise*.5, arm, rise*.5)]
    elif tendency == 5:
        parts += [seg(-arm, -rise*.5, 0, rise*.5), seg(0, rise*.5, arm, -rise*.5)]
    elif tendency == 1:
        parts += [seg(-arm, rise*.5, 0, -rise*.5), seg(0, -rise*.5, arm, -rise*.5)]
    elif tendency == 6:
        parts += [seg(-arm, -rise*.5, 0, rise*.5), seg(0, rise*.5, arm, rise*.5)]
    return ''.join(parts)


def station_model_svg(d, S=34):
    """Full WMO station model SVG."""
    PAD = S * CANVAS_PAD
    W   = S * 3 + PAD * 2
    H   = S * CANVAS_H_FACTOR + PAD * 2
    cx  = W / 2
    cy  = H / 2
    R   = S * CIRCLE_RADIUS
    fs  = max(FONT_MIN_PX, int(S * FONT_SIZE_SCALE))
    off = R + S * LABEL_HORIZ_OFF

    parts = []

    # ── Sky cover / triangle ──────────────────────────────────────────────
    if d.get('has_sky_obs', False):
        parts.append(cloud_circle_svg(cx, cy, R, d['oktas']))
    else:
        th = R * 1.6
        parts.append(
            f'<polygon points="{cx:.2f},{cy-th:.2f} '
            f'{cx-th:.2f},{cy+th*0.65:.2f} '
            f'{cx+th:.2f},{cy+th*0.65:.2f}" '
            f'fill="black" stroke="none"/>'
        )

    # ── Wind barb ─────────────────────────────────────────────────────────
    parts.append(wind_barb_svg(cx, cy, R,
                               d['wind_dir'], d['wind_spd'],
                               d.get('wind_gust', 0), S))

    # ── Text helper ───────────────────────────────────────────────────────
    def txt(x, y, text, anchor='end', size=None):
        sz = size or fs
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{anchor}" dominant-baseline="central" '
            f'font-size="{sz}px" font-weight="bold" '
            f'font-family="Courier New,monospace" fill="black" '
            f'paint-order="stroke" stroke="white" '
            f'stroke-width="2" stroke-linejoin="round">'
            f'{text}</text>'
        )

    # ── Temperature (top-left) ────────────────────────────────────────────
    if d['temp'] is not None:
        parts.append(txt(cx - off, cy - R * 0.6 - LABEL_VERT_OFF, str(d['temp'])))

    # ── Vis + weather (left) ──────────────────────────────────────────────
    v  = d['vis']
    vs = (str(int(v))  if v is not None and v >= 10    else
          str(int(v))  if v is not None and v % 1 == 0 else
          f'{v:.1f}'   if v is not None                else None)
    wx = ' '.join(x for x in [vs, d['weather'] or None] if x)
    if wx:
        parts.append(txt(cx - off - 4, cy, wx))

    # ── Dewpoint (bottom-left) ────────────────────────────────────────────
    if d['dew'] is not None:
        parts.append(txt(cx - off, cy + R * 0.6 + LABEL_VERT_OFF, str(d['dew'])))

    # ── SLP label (top-right) ─────────────────────────────────────────────
    if d['slp_label']:
        parts.append(txt(cx + off, cy - R * 0.6 - LABEL_VERT_OFF,
                         d['slp_label'], anchor='start'))

    # ── Pressure change + tendency symbol (right) ─────────────────────────
    tendency        = d.get('tendency')
    pressure_change = d.get('pressure_change')
    if tendency is not None:
        tend_y     = cy - R * 0.6 - LABEL_VERT_OFF + S * 0.65
        has_number = tendency != 'steady' and pressure_change is not None
        if has_number:
            sign   = '+' if pressure_change > 0 else ('-' if pressure_change < 0 else '')
            pc_str = sign + str(abs(pressure_change))
            parts.append(txt(cx + off, tend_y, pc_str, anchor='start'))
        parts.append(pressure_tendency_svg(cx, cy, R, tendency, S, fs))

    # ── Ceiling height (below circle) ────────────────────────────────────
    if d['lowest_sig'] and d['lowest_sig']['height'] <= 120:
        _cb = math.ceil(d['lowest_sig']['height'] / 10)
        parts.append(txt(cx, cy + R + fs * LABEL_ROW_GAP,
                         str(_cb), anchor='middle'))

    # ── Station ID (bottom) ───────────────────────────────────────────────
    _name_y = cy + R + fs * LABEL_ROW_GAP + fs * (LABEL_ROW_GAP + 0.2)
    parts.append(txt(cx, _name_y, d['icao'][-3:], anchor='middle'))

    return (
        f'<svg width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.2f} {H:.2f}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="overflow:visible">'
        + ''.join(parts)
        + '</svg>'
    ), W, H


def flight_cat_color(d):
    return {
        'VFR':  '#22aa44',
        'MVFR': '#2244cc',
        'IFR':  '#cc2222',
        'LIFR': '#880088',
    }.get(d.get('flt_cat', ''), '#888888')


# ══════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════
print(f'Station model SVG ready  '
      f'(FEATHER_SIDE={FEATHER_SIDE}, FEATHER_ANGLE={FEATHER_ANGLE}°, '
      f'FONT_SIZE_SCALE={FONT_SIZE_SCALE})')

_demo_records = [
    dict(icao='CYWG', name='Winnipeg',  wind_dir=270, wind_spd=25, wind_gust=0,
         temp=15,  dew=8,   vis=15,  weather='',
         slp=1013.2, slp_label='132',
         oktas=4, has_sky_obs=True,
         clouds=[{'cover':'SCT','height':25,'raw':'SCT025'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',
         lat=0, lon=0, timestamp='', rh=60,
         tendency='rising',         pressure_change=+28),

    dict(icao='CYYZ', name='Toronto',   wind_dir=0,   wind_spd=20, wind_gust=0,
         temp=2,   dew=-1,  vis=3,   weather='BR',
         slp=1001.4, slp_label='014',
         oktas=8, has_sky_obs=True,
         clouds=[{'cover':'OVC','height':8,'raw':'OVC008'}],
         lowest_sig={'cover':'OVC','height':8,'raw':'OVC008'},
         ceiling=800, flt_cat='IFR',
         lat=0, lon=0, timestamp='', rh=82,
         tendency='falling',        pressure_change=-15),

    dict(icao='CYVR', name='Vancouver', wind_dir=90,  wind_spd=35, wind_gust=0,
         temp=8,   dew=6,   vis=1.5, weather='-RA',
         slp=998.6, slp_label='986',
         oktas=8, has_sky_obs=True,
         clouds=[{'cover':'OVC','height':4,'raw':'OVC004'}],
         lowest_sig={'cover':'OVC','height':4,'raw':'OVC004'},
         ceiling=400, flt_cat='LIFR',
         lat=0, lon=0, timestamp='', rh=88,
         tendency='steady',         pressure_change=+2),

    dict(icao='CYQF', name='Red Deer',  wind_dir=180, wind_spd=50, wind_gust=0,
         temp=-5,  dew=-12, vis=15,  weather='',
         slp=1020.8, slp_label='208',
         oktas=2, has_sky_obs=True,
         clouds=[{'cover':'FEW','height':40,'raw':'FEW040'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',
         lat=0, lon=0, timestamp='', rh=55,
         tendency='rising_falling', pressure_change=+10),

    dict(icao='CYYC', name='Calgary',   wind_dir=315, wind_spd=65, wind_gust=0,
         temp=-18, dew=-22, vis=9,   weather='SN',
         slp=1008.0, slp_label='080',
         oktas=6, has_sky_obs=True,
         clouds=[{'cover':'BKN','height':15,'raw':'BKN015'}],
         lowest_sig={'cover':'BKN','height':15,'raw':'BKN015'},
         ceiling=1500, flt_cat='MVFR',
         lat=0, lon=0, timestamp='', rh=72,
         tendency='falling_rising', pressure_change=-22),

    dict(icao='CYEG', name='Edmonton',  wind_dir=225, wind_spd=15, wind_gust=0,
         temp=-2,  dew=-8,  vis=15,  weather='',
         slp=1015.4, slp_label='154',
         oktas=2, has_sky_obs=True,
         clouds=[{'cover':'FEW','height':50,'raw':'FEW050'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',
         lat=0, lon=0, timestamp='', rh=62,
         tendency='rising_steady',  pressure_change=+18),

    dict(icao='CYED', name='Namao',     wind_dir=200, wind_spd=10, wind_gust=0,
         temp=-4,  dew=-10, vis=15,  weather='',
         slp=1012.1, slp_label='121',
         oktas=4, has_sky_obs=True,
         clouds=[{'cover':'SCT','height':30,'raw':'SCT030'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',
         lat=0, lon=0, timestamp='', rh=65,
         tendency='falling_steady', pressure_change=-8),
]

_S      = 44
_margin = 8

_svg_parts_list = [station_model_svg(r, S=_S) + (r,) for r in _demo_records]
_cell_w = int(_svg_parts_list[0][1])
_cell_h = int(_svg_parts_list[0][2])
_label_h = 30
_total_w = len(_svg_parts_list) * (_cell_w + _margin) + _margin
_total_h = _cell_h + _label_h

_out = [
    f'<svg width="{_total_w}" height="{_total_h}" '
    f'xmlns="http://www.w3.org/2000/svg" '
    f'style="background:#f8f8f8;font-family:Courier New,monospace">'
]

for _i, (_svg_str, _sw, _sh, _rec) in enumerate(_svg_parts_list):
    _ox = _margin + _i * (_cell_w + _margin)
    _cx = _ox + _cell_w / 2
    _inner = _svg_str.split('>', 1)[1].rsplit('</svg>', 1)[0]
    _out.append(
        f'<rect x="{_ox}" y="0" width="{_cell_w}" height="{_cell_h}" '
        f'fill="white" stroke="#ccc" stroke-width="0.8"/>'
        f'<g transform="translate({_ox},0)">{_inner}</g>'
        f'<text x="{_cx:.1f}" y="{_cell_h+14}" text-anchor="middle" '
        f'font-size="9" fill="#333">'
        f'{_rec["icao"]} {_rec["wind_dir"]:03d}/{_rec["wind_spd"]}kt</text>'
        f'<text x="{_cx:.1f}" y="{_cell_h+24}" text-anchor="middle" '
        f'font-size="8" font-weight="bold" fill="{flight_cat_color(_rec)}">'
        f'{_rec["flt_cat"]}</text>'
    )

_out.append('</svg>')



# ── Cell 3 . Load station list from orangecore.net ────────────
import csv, io, math as _math




# ====================================================
#=====================================================
import threading, time

##########################################################



def load_stations(url, coverage='standard'):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    stations = {}
    for row in reader:
        icao = row.get('Code','').strip()
        if not icao: continue

        if coverage == 'chart':
            chart_keys = [k for k in row.keys() if k.strip().lower() == 'chart']
            chart_val  = row.get(chart_keys[0], '').strip() if chart_keys else ''
            if not chart_val:
                continue
        else:
            tier_map = {'essential': 1, 'standard': 2, 'all': 3}
            max_tier = tier_map.get(coverage, 2)
            tier = (1 if row.get('ESSENTIAL','').strip() else
                    2 if row.get('STANDARD','').strip()  else 3)
            if tier > max_tier:
                continue

        try:
            stations[icao] = {
                'icao': icao,
                'name': row.get('Name','').strip(),
                'lat':  float(row['Latitude']),
                'lon':  float(row['Longitude']),
                'tier': 0 if coverage == 'chart' else tier,
                'source': 'metar',
            }
        except (ValueError, KeyError):
            pass

    return stations

STATIONS = load_stations(CSV_URL, COVERAGE)
print(f'✓ Loaded {len(STATIONS)} stations ({COVERAGE} tier)')

lats = [s['lat'] for s in STATIONS.values()]
lons = [s['lon'] for s in STATIONS.values()]
print(f'  Lat range: {min(lats):.1f}°N - {max(lats):.1f}°N')
print(f'  Lon range: {min(lons):.1f}°E - {max(lons):.1f}°E')


# ── Register EC model virtual stations into STATIONS ─────────────────────
# Injected here so parse_metar_line() and ALL downstream cells see them
# as ordinary stations, indistinguishable from real ones except source='ecmodel'.

EC_LONGITUDE = list(range(-114, -140, 10))
EC_LATITUDES = list(range(50, 40, -10))  # [49, 39, 29, 19, 9] south to equator
OPENMETEO_URL = 'https://api.open-meteo.com/v1/forecast'

def ec_icao(lat, lon):
    return f"ECML{abs(lat):02d}{abs(lon):03d}"

for _lat in EC_LATITUDES:
    for _lon in EC_LONGITUDE:
        _id = ec_icao(_lat, _lon)

        STATIONS[_id] = {
            'icao':   _id,
            'name':   f'EC Model {_lat:+d}N {abs(_lon):.0f}W',
            'lat':    float(_lat),
            'lon':    float(_lon),
            'tier':   0,
            'source': 'ecmodel',
        }

print(f'  + {len(EC_LATITUDES)} EC model virtual stations registered')
print(f'  Total STATIONS: {len(STATIONS)}')

# ── Aliases so UA-2b references resolve to the same objects ──────────────
ec_ua_icao      = ec_icao
EC_UA_LONGITUDE = EC_LONGITUDE
EC_UA_LATITUDES = EC_LATITUDES


# ── Cell 3b . Fetch & parse EC model data from Open-Meteo ─────────────────────
# Produces ec_metar_records[] with the IDENTICAL schema as parse_metar_line().
# Appended to metar_records[] at the END of Cell 5 — before Cell 5b runs.

from datetime import datetime, timezone as _tz



def _ec_rh(temp, dew):
    if temp is None or dew is None: return None
    a, b = 17.625, 243.04
    rh = round(100 * _math.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
    return max(0, min(100, rh))

def _ec_slp_label(slp):
    return '' if slp is None else f'SLP{int(round(slp * 10)) % 1000:03d}'

def _ec_vis(prec):
    if prec is None or prec < 0.1: return 10.0
    if prec < 2.5:  return 5.0
    if prec < 7.6:  return 2.0
    return 0.5

def _ec_wx(prec):
    if prec is None or prec < 0.1: return ''
    if prec < 2.5: return '-RA'
    if prec < 7.6: return 'RA'
    return '+RA'

def _ec_cat(vis, ceil):
    if ceil < 500  or vis < 1: return 'LIFR'
    if ceil < 1000 or vis < 3: return 'IFR'
    if ceil < 3000 or vis < 5: return 'MVFR'
    return 'VFR'

def _ec_tfmt(c):
    if c is None: return '//'
    i = int(round(c))
    return f'M{abs(i):02d}' if i < 0 else f'{i:02d}'

# ── fetch one grid point ──────────────────────────────────────────────────────
def _fetch_ec(lat, lon, past_days=0, forecast_days=3):
    r = requests.get(OPENMETEO_URL, params={
        'latitude': lat, 'longitude': lon,
        'hourly': ('temperature_2m,precipitation,pressure_msl,'
                   'wind_speed_10m,wind_direction_10m,wind_gusts_10m,dew_point_2m'),
        'models': 'ecmwf_ifs',
        'past_days': past_days, 'forecast_days': forecast_days,
        'wind_speed_unit': 'kn', 'timezone': 'UTC',
    }, timeout=20)
    r.raise_for_status()
    return r.json()

# ── parse one response → list of METAR-schema dicts ──────────────────────────
def _parse_ec(lat, lon, data):
    icao   = ec_icao(lat, lon)
    st     = STATIONS[icao]
    hourly = data.get('hourly', {})
    times  = hourly.get('time', [])

    def col(k): return hourly.get(k, [])
    T  = col('temperature_2m');   D  = col('dew_point_2m')
    SL = col('pressure_msl')
    WD = col('wind_direction_10m'); WS = col('wind_speed_10m')
    WG = col('wind_gusts_10m');   PR = col('precipitation')

    records = []
    for i, iso in enumerate(times):
        def g(lst): return lst[i] if i < len(lst) else None
        temp=g(T); dew=g(D); slp=g(SL)
        wdir=g(WD); wspd=g(WS); wgst=g(WG); prec=g(PR)

        # Timestamp → DDHHmmZ (model data is always on the hour)
        try:
            dt = datetime.fromisoformat(iso).replace(tzinfo=_tz.utc)
            ts = dt.strftime('%d%H00Z')
        except Exception:
            ts = '//////Z'

        sky         = 'SKC'
        oktas       = 0
        clouds_list = []
        ceiling     = 99999
        lowest_sig  = None

        vis     = _ec_vis(prec)
        wx      = _ec_wx(prec)
        flt_cat = _ec_cat(vis, ceiling)
        rh      = _ec_rh(temp, dew)
        slp_lbl = _ec_slp_label(slp)

        if wdir is not None and wspd is not None:
            wd, ws = int(round(wdir)), int(round(wspd))
            wg     = int(round(wgst)) if wgst else 0
            gust   = f'G{wg:02d}' if wg > ws + 5 else ''
            wind_g = f'{wd:03d}{ws:02d}{gust}KT'
            wind_gust_out = wg if (wgst and wgst > wspd + 5) else 0
        else:
            wind_g = '/////KT'; wd = ws = wg = None; wind_gust_out = 0

        vis_str   = f'{int(vis)}SM' if vis == int(vis) else f'{vis:.1f}SM'
        metar_str = ' '.join(p for p in [
            'METAR', icao, ts, 'AUTO', wind_g, vis_str, wx,
            sky, f'{_ec_tfmt(temp)}/{_ec_tfmt(dew)}', slp_lbl,
            'RMK ECMWF_IFS',
        ] if p)

        records.append(dict(
            icao=icao, name=st['name'],
            lat=lat, lon=lon,
            source='ecmodel',
            timestamp=ts,
            wind_dir=wdir, wind_spd=wspd, wind_gust=wind_gust_out,
            vis=vis, temp=temp, dew=dew, rh=rh,
            slp=slp, slp_label=slp_lbl,
            has_sky_obs=False, oktas=oktas,
            clouds=clouds_list, lowest_sig=lowest_sig, ceiling=ceiling,
            weather=wx, flt_cat=flt_cat,
            tendency=None, pressure_change=None,
            metar_str=metar_str,
        ))
    return records

# ── fetch loop ────────────────────────────────────────────────────────────────
ec_metar_records = []
ec_fetch_errors  = []

total_points = len(EC_LATITUDES) * len(EC_LONGITUDE)

print(f"Fetching {total_points} EC model grid points ({len(EC_LATITUDES)} lat × {len(EC_LONGITUDE)} lon)...")

for _lat in EC_LATITUDES:
    for _lon in EC_LONGITUDE:

        _id = ec_icao(_lat, _lon)
        print(f'  {_id}  ({_lat:+03d}°, {_lon}°) … ', end='')

        try:
            data = _fetch_ec(_lat, _lon)
            _recs = _parse_ec(_lat, _lon, data)

            ec_metar_records.extend(_recs)
            print(f'✓ {len(_recs)} hourly obs')

        except Exception as _e:
            print(f'✗ {_e}')
            ec_fetch_errors.append(_id)


# ── Cell 4 . Fetch live METARs from aviationweather.gov ───────
import concurrent.futures, time

EXPECTED_HOURS = []

def fetch_chunk(codes, hours=12, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            params = {'ids': ','.join(codes), 'format': 'raw',
                      'hours': hours, 'mostRecent': 'false'}
            r = requests.get(METAR_API, params=params, timeout=30)
            if r.ok and r.text.strip():
                return r.text, []
            time.sleep(backoff * (attempt + 1))
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return '', codes

def fetch_all_metars(station_codes, chunk_size=25, max_workers=6, hours=12):
    chunks = [station_codes[i:i+chunk_size]
              for i in range(0, len(station_codes), chunk_size)]
    chunk_lines = ''.join(
        f'<div style="font-family:monospace;font-size:11px;color:#555;margin:1px 0">'
        f'Chunk {i+1}: {", ".join(c)}</div>'
        for i, c in enumerate(chunks)
    )

    raw_parts = []; failed_codes = []; done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_chunk, c, hours): c for c in chunks}
        for fut in concurrent.futures.as_completed(futures):
            text, failed = fut.result()
            if text:  raw_parts.append(text)
            if failed: failed_codes.extend(failed)
            done += len(futures[fut])
            print(f'  {done}/{len(station_codes)} ({int(done/len(station_codes)*100)}%)', end='\r')
    print()

    joined = '\n'.join(raw_parts)
    seen_icaos = set()
    for line in joined.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            icao = parts[1] if parts[0] in ('METAR','SPECI') else parts[0]
            seen_icaos.add(icao)

    silent_missing = [s for s in station_codes
                      if s not in seen_icaos and s not in failed_codes]
    if silent_missing:
        print(f'  ↻ Pass 2: retrying {len(silent_missing)} silent-missing stations...')
        retry_chunks = [silent_missing[i:i+chunk_size]
                        for i in range(0, len(silent_missing), chunk_size)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures2 = {ex.submit(fetch_chunk, c, hours, 2, 1): c for c in retry_chunks}
                for fut in concurrent.futures.as_completed(futures2, timeout=10):
                    text, failed = fut.result()
                    if text:  raw_parts.append(text)
                    if failed: failed_codes.extend(failed)
            print('  ↻ Pass 2 done.')
        except concurrent.futures.TimeoutError:
            print(f'  ↻ Pass 2 timed out — skipping {len(silent_missing)} stations')

    return '\n'.join(raw_parts), failed_codes


# Only send real station codes to aviationweather.gov (not EC virtual stations)
real_codes = [c for c in STATIONS if STATIONS[c].get('source') != 'ecmodel']
raw_metar_text, failed_chunks = fetch_all_metars(real_codes, hours=12)
line_count = sum(1 for l in raw_metar_text.splitlines() if l.strip())
print(f'✓ Fetched {line_count} raw METAR lines')

returned_icaos = set()
for line in raw_metar_text.splitlines():
    parts = line.strip().split()
    if len(parts) >= 2:
        icao = parts[1] if parts[0] in ('METAR', 'SPECI') else parts[0]
        returned_icaos.add(icao)

no_data = [s for s in real_codes if s not in returned_icaos]

warnings = []
if failed_chunks:
    warnings.append(f"CHUNK FETCH FAILED — {len(failed_chunks)} stations lost:<br>"
                    + "&nbsp;&nbsp;" + "&nbsp;&nbsp;".join(failed_chunks))
if no_data:
    warnings.append(f"NO DATA RETURNED for {len(no_data)} stations:<br>"
                    + "&nbsp;&nbsp;" + "&nbsp;&nbsp;".join(no_data))

if warnings:
    pass
else:
    pass


# ── Cell 5 . Parse METAR fields ───────────────────────────────

def parse_metar_line(line, stations):
    '''Parse one METAR line → dict or None'''
    parts = line.strip().split()
    if len(parts) < 5: return None
    idx = 0
    if parts[0] == 'SPECI': return None
    if parts[0] == 'METAR': idx = 1
    if idx >= len(parts): return None
    icao = parts[idx]
    if icao not in stations: return None
    st = stations[icao]

    ts_raw = parts[idx+1] if idx+1 < len(parts) else ''
    if not re.match(r'^\d{6}Z$', ts_raw): return None
    day, hour, minute = int(ts_raw[0:2]), int(ts_raw[2:4]), int(ts_raw[4:6])
    if minute >= 35:
        hour = (hour + 1) % 24; minute = 0
    elif minute <= 25:
        minute = 0
    else:
        return None
    timestamp = f'{day:02d}{hour:02d}00Z'

    rest = parts[idx+2:]
    rest = [p for p in rest if p not in ('MISG', 'MSIG')]

    wind_dir = wind_spd = wind_gust = None
    for p in rest:
        m = re.match(r'^(\d{3})(\d{2,3})(?:G(\d{2,3}))?KT$', p)
        if m:
            wind_dir, wind_spd = int(m[1]), int(m[2])
            wind_gust = int(m[3]) if m[3] else 0
            break
        if re.match(r'^00000KT$', p): wind_dir=0; wind_spd=0; wind_gust=0; break

    vis = None
    for i, p in enumerate(rest):
        if p.endswith('SM'):
            whole = int(rest[i-1]) if i > 0 and rest[i-1].isdigit() else 0
            frac_str = p[:-2].lstrip('M')
            if '/' in frac_str:
                try:
                    n, d = frac_str.split('/')
                    vis = whole + int(n) / int(d)
                except (ValueError, ZeroDivisionError):
                    vis = 0.0
            else:
                try: vis = whole + float(frac_str) if frac_str else float(whole)
                except: vis = None
            break

    cloud_re = re.compile(r'^(FEW|SCT|BKN|OVC|VV)(\d{3})')
    clouds = []
    for p in rest:
        m = cloud_re.match(p)
        if m: clouds.append({'cover': m[1], 'height': int(m[2]), 'raw': p})
    clouds.sort(key=lambda c: c['height'])
    clr = any(p in ('CLR', 'SKC', 'CAVOK') for p in rest)
    has_sky_obs = clr or bool(clouds)
    cover_rank = {'CLR':0,'SKC':0,'FEW':2,'SCT':4,'BKN':6,'OVC':8,'VV':9}
    oktas = 0 if (clr or not clouds) else max(cover_rank.get(c['cover'], 0) for c in
                  ([c for c in clouds if c['cover'] in ('BKN','OVC','VV')] or clouds))
    sig_clouds = [c for c in clouds if c['cover'] in ('BKN','OVC','VV')]
    ceiling    = sig_clouds[0]['height'] * 100 if sig_clouds else 99999
    lowest_sig = sig_clouds[0] if sig_clouds else None

    temp = dew = None
    for p in rest:
        m = re.match(r'^(M?\d{1,2})/(M?\d{1,2})$', p)
        if m:
            def td(s): return -(int(s[1:])) if s.startswith('M') else int(s)
            temp, dew = td(m[1]), td(m[2])
            break

    slp = None
    for p in rest:
        m = re.match(r'^SLP(\d{3})$', p)
        if m:
            v = int(m[1])
            slp = (900 + v/10) if v >= 500 else (1000 + v/10)
            break

    wx_re = re.compile(
        r'^[+-]?(FZ|SH|BL|TS|MI|PR|BC|DR)?'
        r'(DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|HZ|FU|VA|DU|SA|SQ|PO|FC|SS|DS){1,3}$'
    )
    wx_parts = [p for p in rest
                if wx_re.match(p)
                and not re.match(r'^(RMK|SLP|AUTO|COR|AO\d)', p)]
    weather = ' '.join(wx_parts)

    rh = None
    if temp is not None and dew is not None:
        a, b = 17.625, 243.04
        rh = round(100 * np.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
        rh = max(0, min(100, rh))

    fc_vis = vis if vis is not None else 99
    if   ceiling < 500  or fc_vis < 1: flt_cat = 'LIFR'
    elif ceiling < 1000 or fc_vis < 3: flt_cat = 'IFR'
    elif ceiling < 3000 or fc_vis < 5: flt_cat = 'MVFR'
    else:                               flt_cat = 'VFR'

    slp_label = f'{int(round(slp*10))%1000:03d}' if slp else ''

    return dict(
        icao=icao, name=st['name'],
        lat=st['lat'], lon=st['lon'],
        source='metar',
        timestamp=timestamp,
        wind_dir=wind_dir, wind_spd=wind_spd, wind_gust=wind_gust,
        vis=vis, temp=temp, dew=dew, rh=rh, slp=slp, slp_label=slp_label,
        has_sky_obs=has_sky_obs, oktas=oktas, clouds=clouds, lowest_sig=lowest_sig,
        ceiling=ceiling, weather=weather, flt_cat=flt_cat,
        tendency=None, pressure_change=None,
        metar_str=line.strip(),
    )


def parse_all(text, stations):
    results = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if not re.match(r'^(METAR |SPECI |[A-Z]{4} \d{6}Z)', line): continue
        if line.startswith(('MISG', 'MSIG', 'SIGMET', 'AIRMET', 'PIREP', 'ATIS')): continue
        d = parse_metar_line(line, stations)
        if d:
            key = (d['icao'], d['timestamp'])
            if key not in seen:
                seen.add(key)
                results.append(d)
    return results


metar_records = parse_all(raw_metar_text, STATIONS)

# ── CYMJ/CYYN mutual exclusion ────────────────────────────────────────────────
_cymj_times = set(d['timestamp'] for d in metar_records if d['icao'] == 'CYMJ')
if _cymj_times:
    _before = len(metar_records)
    metar_records = [d for d in metar_records if not (d['icao'] == 'CYYN')]
    print(f'  CYMJ present — removed CYYN ({_before - len(metar_records)} records dropped)')
else:
    print(f'  CYMJ not available — keeping CYYN')
# ── CZPC/CYQL mutual exclusion ────────────────────────────────────────────────
_czpc_times = set(d['timestamp'] for d in metar_records if d['icao'] == 'CZPC')
if _czpc_times:
    _before = len(metar_records)
    metar_records = [d for d in metar_records if not (d['icao'] == 'CYQL')]
    print(f'  CZPC present — removed CYQL ({_before - len(metar_records)} records dropped)')
else:
    print(f'  CZPC not available — keeping CYQL')
# ──────────────────────────────────────────────────────────────────────────────






# ── Merge EC model records ────────────────────────────────────────────────────
# Done here, BEFORE the timestep check and BEFORE Cell 5b, so tendency is
# computed identically for both real and model data.
_n_real = len(metar_records)
_metar_timestamps = set(d['timestamp'] for d in metar_records)
ec_metar_records  = [d for d in ec_metar_records if d['timestamp'] in _metar_timestamps]
metar_records     = metar_records + ec_metar_records
print(f'✓ Merged: {_n_real} real METARs + {len(ec_metar_records)} EC obs = {len(metar_records)} total')

# ── Warn on missing timesteps ─────────────────────────────────────────────────
from collections import defaultdict

all_timestamps = sorted(set(d['timestamp'] for d in metar_records))
station_times  = defaultdict(set)
for d in metar_records:
    station_times[d['icao']].add(d['timestamp'])

missing = {}
for icao in station_times:
    gaps = [ts for ts in all_timestamps if ts not in station_times[icao]]
    if gaps:
        missing[icao] = gaps

total_missing = sum(len(v) for v in missing.values())

slp_count  = sum(1 for d in metar_records if d['slp'])
wind_count = sum(1 for d in metar_records if d['wind_dir'] is not None)
temp_count = sum(1 for d in metar_records if d['temp'] is not None)

if missing:
    rows = ''.join(
        f'<tr><td style="padding:3px 14px;color:#5a3a00;font-family:monospace;">{icao}</td>'
        f'<td style="padding:3px 14px;color:#7a5000;font-family:monospace;text-align:center;">{len(gaps)}</td>'
        f'<td style="padding:3px 14px;color:#7a5000;font-family:monospace;">{", ".join(gaps)}</td></tr>'
        for icao, gaps in sorted(missing.items())
    )
    _all_stations  = sorted(set(d['icao'] for d in metar_records))
    _good_stations = sorted(set(d['icao'] for d in metar_records) - set(missing.keys()))
    # Build latest record per station for tooltip
    _latest = {}
    for d in metar_records:
        if d['icao'] not in _latest or d['timestamp'] > _latest[d['icao']]['timestamp']:
            _latest[d['icao']] = d

    def _station_badge(icao):
        d = _latest.get(icao, {})
        has_gap  = icao in missing
        bg       = '#fff3b0' if has_gap else '#e6faf0'
        bdr      = '#e6a800' if has_gap else '#1a7a3a'
        clr      = '#7a5000' if has_gap else '#145c2c'
        temp_str = f"{d.get('temp','—')}°C" if d.get('temp') is not None else '—'
        dew_str  = f"{d.get('dew','—')}°C"  if d.get('dew')  is not None else '—'
        slp_str  = f"{d.get('slp','—')} hPa" if d.get('slp') is not None else '—'
        wdir     = d.get('wind_dir')
        wspd     = d.get('wind_spd')
        wind_str = f"{wdir}°/{wspd}kt" if wdir is not None and wspd is not None else '—'
        cat      = d.get('flt_cat', '—')
        ts       = d.get('timestamp', '—')
        src      = d.get('source', '—')
        gap_str  = f"⚠ missing: {', '.join(missing[icao])}" if has_gap else '✔ complete'
        detail_id = f'stn-detail-{icao}'
        popup_bdr = '#a85c00' if has_gap else '#1a7a3a'
        popup_gap_clr = '#a85c00' if has_gap else '#1a7a3a'
        # All METAR lines for this station, sorted by timestamp
        all_lines = [
            r for r in metar_records if r['icao'] == icao
        ]
        all_lines.sort(key=lambda r: r['timestamp'])
        metar_rows = ''.join(
            f'<tr style="border-bottom:1px solid #eee;">'
            f'<td style="padding:2px 8px;color:#555;white-space:nowrap;">{r["timestamp"]}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;font-size:10px;'
            f'color:#1a2030;white-space:nowrap;">{r.get("metar_str","—")}</td>'
            f'</tr>'
            for r in all_lines
        )

        return (
            f'<span style="display:inline-block;position:relative;margin:2px;">'
            f'<span onclick="'
            f'var p=document.getElementById(\'{detail_id}\');'
            f'document.querySelectorAll(\'.stn-detail-popup\').forEach(function(x){{if(x.id!==\'{detail_id}\')x.style.display=\'none\';}});'
            f'p.style.display=p.style.display===\'none\'?\'block\':\'none\';" '
            f'style="font-family:monospace;font-size:11px;color:{clr};cursor:pointer;'
            f'background:{bg};border:1px solid {bdr};border-radius:3px;'
            f'padding:1px 6px;display:inline-block;">{icao}</span>'
            f'<div id="{detail_id}" class="stn-detail-popup" '
            f'style="display:none;position:absolute;top:20px;left:0;z-index:9999;'
            f'background:#fff;border:2px solid {popup_bdr};border-radius:8px;padding:12px 16px;'
            f'font-family:monospace;font-size:12px;color:#1a2030;'
            f'box-shadow:0 4px 16px rgba(0,0,0,0.25);min-width:420px;max-width:700px;">'
            f'<b style="font-size:13px;color:#1a2030;">{icao}</b> '
            f'<span style="color:#888;font-size:10px;">{d.get("name","")}</span> '
            f'<span style="color:#888;font-size:10px;">· {src} · {len(all_lines)} obs</span>'
            f'<hr style="margin:4px 0;border:none;border-top:1px solid #ccc;">'
            f'<span style="color:{popup_gap_clr};font-size:10px;">{gap_str}</span>'
            f'<hr style="margin:4px 0;border:none;border-top:1px solid #eee;">'
            f'<div style="max-height:300px;overflow-y:auto;">'
            f'<table style="border-collapse:collapse;width:100%;font-size:10px;">'
            f'<tr style="background:#f0f4f8;"><th style="padding:2px 8px;text-align:left;">Time</th>'
            f'<th style="padding:2px 8px;text-align:left;">METAR</th></tr>'
            f'{metar_rows}'
            f'</table>'
            f'</div>'
            f'<button onclick="document.getElementById(\'{detail_id}\').style.display=\'none\';event.stopPropagation();" '
            f'style="margin-top:8px;font-size:10px;padding:2px 10px;cursor:pointer;'
            f'border:1px solid #aaa;border-radius:3px;background:#f0f0f0;">✕ close</button>'
            f'</div>'
            f'</span>'
        )

    _good_rows = ''.join(_station_badge(icao) for icao in _all_stations)
else:
    pass


print(f'  SLP: {slp_count}  Wind: {wind_count}  Temp: {temp_count}')

# ── Summary table ─────────────────────────────────────────────────────────────
import pandas as pd

_df = pd.DataFrame([{
    'ICAO':       d['icao'],
    'Src':        d.get('source', 'metar'),
    'Name':       d['name'],
    'Time':       d['timestamp'],
    'Lat':        d['lat'],
    'Lon':        d['lon'],
    'Temp(C)':    d['temp'],
    'Dew(C)':     d['dew'],
    'RH(%)':      d['rh'],
    'Wind Dir':   d['wind_dir'],
    'Wind Spd':   d['wind_spd'],
    'Wind Gust':  d['wind_gust'],
    'Vis(SM)':    d['vis'],
    'Wx':         d['weather'],
    'Oktas':      d['oktas'],
    'Ceiling':    d['ceiling'],
    'SLP(hPa)':   d['slp'],
    'SLP Lbl':    d['slp_label'],
    'Tendency':   d.get('tendency'),
    'P Change':   d.get('pressure_change'),
    'Sky Obs':    d['has_sky_obs'],
    'Lowest Sig': d['lowest_sig']['raw'] if d['lowest_sig'] else None,
    'Clouds':     ' '.join(c['raw'] for c in d['clouds']),
    'Cat':        d['flt_cat'],
} for d in metar_records])

def _style_df(df, caption):
    grad_cols = [c for c in ['Temp(C)','Dew(C)'] if c in df.columns]
    slp_cols  = [c for c in ['SLP(hPa)']          if c in df.columns]
    okta_cols = [c for c in ['Oktas']              if c in df.columns]
    s = df.style.set_caption(caption)
    if grad_cols:  s = s.background_gradient(subset=grad_cols, cmap='RdYlBu_r')
    if slp_cols:   s = s.background_gradient(subset=slp_cols,  cmap='coolwarm')
    if okta_cols:  s = s.background_gradient(subset=okta_cols, cmap='Greys')
    s = s.map(lambda v: (
        'color:red;font-weight:bold' if v == 'LIFR' else
        'color:crimson'              if v == 'IFR'  else
        'color:steelblue'            if v == 'MVFR' else
        'color:green'                if v == 'VFR'  else ''), subset=['Cat'])
    s = s.map(lambda v: 'background:#ddeeff;font-style:italic' if v == 'ecmodel' else '',
                   subset=['Src'])
    s = s.format(na_rep='—', precision=1)
    return s.to_html()

_ROWS = 5
_uid  = 'metartbl'
_short_html = _style_df(_df.head(_ROWS), f'METARs + EC Model — showing {_ROWS} of {len(_df)} records')
_full_html  = _style_df(_df,             f'METARs + EC Model — {len(_df)} records total')



# ── Cell 5b . Compute pressure tendency from 3-hr SLP history ─
# UNCHANGED — now naturally covers both real METAR and EC model records.

from collections import defaultdict

def classify_tendency(slp_now, slp_3h):
    if slp_now is None or slp_3h is None:
        return None, None
    diff   = slp_now - slp_3h
    change = int(round(diff * 10))
    if abs(diff) < 1.0: return 'steady', change
    return ('rising', change) if diff > 0 else ('falling', change)

def classify_tendency_detailed(slp_series):
    if len(slp_series) < 2: return None, None
    slp_vals = [s for _, s in slp_series if s is not None]
    if len(slp_vals) < 2: return None, None
    first = slp_vals[0]; last = slp_vals[-1]; mid = slp_vals[len(slp_vals)//2]
    diff_total = last - first; diff_first = mid - first; diff_last = last - mid
    STEADY = 1.0
    change = int(round(diff_total * 10))
    def sign(x): return 1 if x > STEADY else (-1 if x < -STEADY else 0)
    s1, s2 = sign(diff_first), sign(diff_last)
    if   s1 ==  1 and s2 ==  1: return 'rising',         change
    elif s1 == -1 and s2 == -1: return 'falling',        change
    elif s1 ==  0 and s2 ==  0: return 'steady',         change
    elif s1 ==  1 and s2 == -1: return 'rising_falling', change
    elif s1 == -1 and s2 ==  1: return 'falling_rising', change
    elif s1 ==  1 and s2 ==  0: return 'rising_steady',  change
    elif s1 == -1 and s2 ==  0: return 'falling_steady', change
    elif s1 ==  0 and s2 ==  1: return 'rising',         change
    elif s1 ==  0 and s2 == -1: return 'falling',        change
    else:                        return 'steady',         change

station_slp_series = defaultdict(list)
for d in metar_records:
    if d['slp'] is not None:
        station_slp_series[d['icao']].append((d['timestamp'], d['slp']))
for icao in station_slp_series:
    station_slp_series[icao].sort(key=lambda x: x[0])

tendency_assigned = 0
for d in metar_records:
    series = [(ts, slp) for ts, slp in station_slp_series[d['icao']]
              if ts <= d['timestamp']]
    if len(series) >= 2:
        tend, change = classify_tendency_detailed(series)
        d['tendency']        = tend
        d['pressure_change'] = change
        tendency_assigned   += 1

print(f'✓ Tendency computed for {tendency_assigned} / {len(metar_records)} records')
no_tend = sum(1 for d in metar_records if d['tendency'] is None)
print(f'  No tendency (insufficient history): {no_tend}')

from collections import Counter
tend_counts = Counter(d['tendency'] for d in metar_records if d['tendency'])
for k, v in sorted(tend_counts.items(), key=lambda x: -x[1]):
    print(f'  {k:<20} {v}')

src_counts = Counter(d.get('source','metar') for d in metar_records)
print(f'\n  Source breakdown in metar_records:')
for src, cnt in src_counts.items():
    print(f'    {src:<10} {cnt} records')

# ── Interactive station badge grid ────────────────────────────
_all_stations_5b  = sorted(set(d['icao'] for d in metar_records))
_no_tend_stations = set(d['icao'] for d in metar_records if d['tendency'] is None)
_good_count_5b    = len(_all_stations_5b) - len(_no_tend_stations)

_latest_5b = {}
for d in metar_records:
    if d['icao'] not in _latest_5b or d['timestamp'] > _latest_5b[d['icao']]['timestamp']:
        _latest_5b[d['icao']] = d

def _station_badge_5b(icao):
    d         = _latest_5b.get(icao, {})
    has_gap   = icao in _no_tend_stations
    bg        = '#fff3b0' if has_gap else '#e6faf0'
    bdr       = '#e6a800' if has_gap else '#1a7a3a'
    clr       = '#7a5000' if has_gap else '#145c2c'
    ts        = d.get('timestamp', '—')
    tend      = d.get('tendency', '—') or '—'
    src       = d.get('source', '—')
    gap_str   = '⚠ no tendency (insufficient history)' if has_gap else '✔ tendency computed'
    detail_id = f'tend-detail-{icao}'
    popup_bdr = '#a85c00' if has_gap else '#1a7a3a'
    popup_gap_clr = '#a85c00' if has_gap else '#1a7a3a'
    all_lines = sorted([r for r in metar_records if r['icao'] == icao],
                       key=lambda r: r['timestamp'])
    metar_rows = ''.join(
        f'<tr style="border-bottom:1px solid #eee;">'
        f'<td style="padding:2px 8px;color:#555;white-space:nowrap;">{r["timestamp"]}</td>'
        f'<td style="padding:2px 8px;font-family:monospace;font-size:10px;color:#1a2030;white-space:nowrap;">'
        f'SLP:{r.get("slp","—")} &nbsp; tend:{r.get("tendency","—")}</td>'
        f'</tr>'
        for r in all_lines
    )
    # Build SLP chart data for this station
    slp_points = [(r["timestamp"], r["slp"]) for r in all_lines if r.get("slp") is not None]
    chart_labels = [p[0] for p in slp_points]
    chart_values = [p[1] for p in slp_points]
    chart_id = f'slp-chart-{icao}'
    chart_labels_js = str(chart_labels).replace("'", '"')
    chart_values_js = str(chart_values)
    return (
        f'<span style="display:inline-block;position:relative;margin:2px;">'
        f'<span onclick="'
        f'var p=document.getElementById(\'{detail_id}\');'
        f'document.querySelectorAll(\'.tend-detail-popup\').forEach(function(x){{if(x.id!==\'{detail_id}\')x.style.display=\'none\';}});'
        f'p.style.display=p.style.display===\'none\'?\'block\':\'none\';" '
        f'style="font-family:monospace;font-size:11px;color:{clr};cursor:pointer;'
        f'background:{bg};border:1px solid {bdr};border-radius:3px;'
        f'padding:1px 6px;display:inline-block;">{icao}</span>'
        f'<div id="{detail_id}" class="tend-detail-popup" '
        f'style="display:none;position:absolute;top:20px;left:0;z-index:9999;'
        f'background:#fff;border:2px solid {popup_bdr};border-radius:8px;padding:12px 16px;'
        f'font-family:monospace;font-size:12px;color:#1a2030;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.25);min-width:320px;max-width:600px;">'
        f'<b style="font-size:13px;">{icao}</b> '
        f'<span style="color:#888;font-size:10px;">{d.get("name","")}</span> '
        f'<span style="color:#888;font-size:10px;">· {src} · {len(all_lines)} obs</span>'
        f'<hr style="margin:4px 0;border:none;border-top:1px solid #ccc;">'
        f'<span style="color:{popup_gap_clr};font-size:10px;">{gap_str}</span>'
        f' &nbsp; <span style="font-size:10px;">latest: {ts} &nbsp; tend: {tend}</span>'
        f'<hr style="margin:4px 0;border:none;border-top:1px solid #eee;">'
        f'<canvas id="{chart_id}" width="460" height="160" '
        f'style="width:100%;max-width:460px;height:160px;margin:8px 0;display:block;"></canvas>'
        f'<script>'
        f'(function(){{'
        f'  var labels = {chart_labels_js};'
        f'  var values = {chart_values_js};'
        f'  var ctx = document.getElementById("{chart_id}");'
        f'  if (!ctx) return;'
        f'  var mn = Math.min.apply(null,values)-1, mx = Math.max.apply(null,values)+1;'
        f'  new Chart(ctx, {{'
        f'    type:"line",'
        f'    data:{{'
        f'      labels:labels,'
        f'      datasets:[{{'
        f'        label:"SLP (hPa)",'
        f'        data:values,'
        f'        borderColor:"#1a4a8a",'
        f'        backgroundColor:"rgba(26,74,138,0.08)",'
        f'        pointBackgroundColor:"#1a4a8a",'
        f'        pointRadius:4,'
        f'        borderWidth:2,'
        f'        tension:0.3,'
        f'        fill:true'
        f'      }}]'
        f'    }},'
        f'    options:{{'
        f'      responsive:false,'
        f'      plugins:{{legend:{{display:false}},'
        f'        tooltip:{{callbacks:{{label:function(c){{return c.parsed.y.toFixed(1)+" hPa";}}}}}}}},'
        f'      scales:{{'
        f'        x:{{ticks:{{font:{{size:9}},maxRotation:45}}}},'
        f'        y:{{min:mn,max:mx,ticks:{{font:{{size:9}}}},title:{{display:true,text:"hPa",font:{{size:9}}}}}}'
        f'      }}'
        f'    }}'
        f'  }});'
        f'}})()'
        f'</script>'
        f'<div style="max-height:160px;overflow-y:auto;">'
        f'<table style="border-collapse:collapse;width:100%;font-size:10px;">'
        f'<tr style="background:#f0f4f8;">'
        f'<th style="padding:2px 8px;text-align:left;">Time</th>'
        f'<th style="padding:2px 8px;text-align:left;">SLP / Tendency</th></tr>'
        f'{metar_rows}</table></div>'
        f'<button onclick="document.getElementById(\'{detail_id}\').style.display=\'none\';event.stopPropagation();" '
        f'style="margin-top:8px;font-size:10px;padding:2px 10px;cursor:pointer;'
        f'border:1px solid #aaa;border-radius:3px;background:#f0f0f0;">✕ close</button>'
        f'</div></span>'
    )

_badge_rows_5b = ''.join(_station_badge_5b(icao) for icao in _all_stations_5b)
_all_ts_5b     = sorted(set(d['timestamp'] for d in metar_records))
_slp_5b        = sum(1 for d in metar_records if d['slp'])
_wind_5b       = sum(1 for d in metar_records if d['wind_dir'] is not None)
_temp_5b       = sum(1 for d in metar_records if d['temp'] is not None)


# ── Cell 5c . Fetch Fort Vermillion (71024) from ogimet ───────────────────────
import requests, re
from datetime import datetime, timezone as _tz

OGIMET_SYNOP_URL = 'https://www.ogimet.com/cgi-bin/getsynop'
FV_WMO   = '71024'
FV_ICAO  = 'CXFV'   # synthetic key — not a real ICAO, but unique in STATIONS
FV_LAT   = 58.3822
FV_LON   = -116.0400
FV_NAME  = 'Fort Vermillion, Alta'

# Register into STATIONS so downstream cells see it
STATIONS[FV_ICAO] = {
    'icao':   FV_ICAO,
    'name':   FV_NAME,
    'lat':    FV_LAT,
    'lon':    FV_LON,
    'tier':   0,
    'source': 'synop',
}

def fetch_ogimet_synop(wmo_id, ndays=2):
    now = datetime.now(_tz.utc)
    # getsynop needs begin/end dates explicitly
    from datetime import timedelta
    start = now - timedelta(days=ndays)
    r = requests.get('https://www.ogimet.com/cgi-bin/getsynop', params={
        'block': wmo_id,
        'begin': start.strftime('%Y%m%d%H%M'),
        'end':   now.strftime('%Y%m%d%H%M'),
    }, timeout=20)
    r.raise_for_status()
    return r.text

print(fetch_ogimet_synop(FV_WMO)[:1000])

def parse_synop_fm12(line, icao, st):
    m = re.match(r'^\d+,(\d{4}),(\d{2}),(\d{2}),(\d{2}),(\d{2}),(.*)', line)
    if not m:
        return None

    dd, hh = int(m.group(4)), int(m.group(5))  # day, hour — already correct
    # wait — CSV cols are: WMO,YYYY,MM,DD,HH,mm,synop
    yyyy, mo, dd, hh = m.group(1), m.group(2), int(m.group(4)), int(m.group(4))
    # re-extract cleanly
    parts_csv = line.split(',', 6)
    if len(parts_csv) < 7:
        return None
    dd   = int(parts_csv[3])
    hh   = int(parts_csv[4])
    synop_str = parts_csv[6].strip()

    timestamp = f'{dd:02d}{hh:02d}00Z'
    _metar_ts_set = set(d['timestamp'] for d in metar_records)
    if timestamp not in _metar_ts_set:
        return None

    groups = synop_str.replace('=', ' ').split()
    # Find WMO index position and start after it
    try:
        data_start = next(i for i, g in enumerate(groups) if g == '71024') + 1
    except StopIteration:
        data_start = 3
    groups = groups[data_start:]

    temp = dew = slp = wind_dir = wind_spd = None

    for g in groups:
        # Wind group: Nddff or /ddff — N or / then 4 digits
        if re.match(r'^[\d/]\d{4}$', g) and wind_dir is None:
            try:
                dd_ = int(g[1:3]) * 10   # tens of degrees → degrees
                ff  = int(g[3:5])
                if 0 < dd_ <= 360:
                    wind_dir = dd_
                    wind_spd = ff
            except: pass

        # 1sTTT — air temperature  (s=0 positive, s=1 negative)
        elif re.match(r'^1[01]\d{3}$', g):
            try:
                sign = -1 if g[1] == '1' else 1
                temp = sign * int(g[2:]) / 10
            except: pass

        # 2sTTT — dew point  (s=0 positive, s=1 negative)
        elif re.match(r'^2[01]\d{3}$', g):
            try:
                sign = -1 if g[1] == '1' else 1
                dew = sign * int(g[2:]) / 10
            except: pass

        # 3PPPP — station pressure (skip)
        elif re.match(r'^3\d{4}$', g):
            pass

        # 4PPPP — sea level pressure
        elif re.match(r'^4\d{4}$', g):
            try:
                raw = int(g[1:])
                # WMO rule: raw is tenths of hPa with leading digit dropped
                # Values 8600–9999 (raw 8600-9999) → 860.0–999.9 hPa
                # Values 0000–8599 (raw 0000-8599) → 1000.0–1085.9 hPa (prepend 10)
                # Threshold: raw >= 8600 means the 9xx range
                slp = (raw / 10.0) if raw >= 8600 else (1000.0 + raw / 10.0)
            except: pass

    rh = None
    if temp is not None and dew is not None:
        import math as _m
        a, b = 17.625, 243.04
        rh = round(100 * _m.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
        rh = max(0, min(100, rh))

    slp_label = f'{int(round(slp*10))%1000:03d}' if slp else ''

    return dict(
        icao=icao, name=st['name'],
        lat=st['lat'], lon=st['lon'],
        source='synop',
        timestamp=timestamp,
        wind_dir=wind_dir, wind_spd=wind_spd, wind_gust=0,
        vis=10.0, temp=temp, dew=dew, rh=rh,
        slp=slp, slp_label=slp_label,
        has_sky_obs=False, oktas=0,
        clouds=[], lowest_sig=None, ceiling=99999,
        weather='', flt_cat='VFR',
        tendency=None, pressure_change=None,
        metar_str=line.strip(),
    )

# ── Fetch and parse ───────────────────────────────────────────────────────────
print(f'Fetching Fort Vermillion (WMO {FV_WMO}) from ogimet...')
fv_records = []
try:
    raw = fetch_ogimet_synop(FV_WMO, ndays=2)
    for line in raw.splitlines():
        if 'AAXX' not in line:
            continue
        rec = parse_synop_fm12(line, FV_ICAO, STATIONS[FV_ICAO])
        if rec:
            fv_records.append(rec)

    # Deduplicate by timestamp — keep latest
    seen = {}
    for r in fv_records:
        seen[r['timestamp']] = r
    fv_records = list(seen.values())

    metar_records.extend(fv_records)
    print(f'✓ Fort Vermillion: {len(fv_records)} obs added → timestamps: {[r["timestamp"] for r in fv_records]}')

except Exception as e:
    print(f'✗ Fort Vermillion fetch failed: {e}')

import pandas as pd

_fv_df = pd.DataFrame([{
    'Timestamp':  r['timestamp'],
    'Temp(C)':    r['temp'],
    'Dew(C)':     r['dew'],
    'RH(%)':      r['rh'],
    'SLP(hPa)':   r['slp'],
    'Wind Dir':   r['wind_dir'],
    'Wind Spd':   r['wind_spd'],
    'Flt Cat':    r['flt_cat'],
    'Source':     r['source'],
} for r in fv_records])

if _fv_df.empty:
    pass
else:
    grad_cols = [c for c in ['Temp(C)','Dew(C)'] if c in _fv_df.columns]
    slp_cols  = [c for c in ['SLP(hPa)']         if c in _fv_df.columns]
    _s = _fv_df.style.set_caption(f'Fort Vermillion (71024 / {FV_ICAO}) — {len(fv_records)} obs')
    if grad_cols: _s = _s.background_gradient(subset=grad_cols, cmap='RdYlBu_r')
    if slp_cols:  _s = _s.background_gradient(subset=slp_cols,  cmap='coolwarm')
    _s = _s.format(na_rep='—', precision=1)


#####################################################

# ── Cell 5b . Compute pressure tendency from 3-hr SLP history ─
# UNCHANGED — now naturally covers both real METAR and EC model records.

from collections import defaultdict

def classify_tendency(slp_now, slp_3h):
    if slp_now is None or slp_3h is None:
        return None, None
    diff   = slp_now - slp_3h
    change = int(round(diff * 10))
    if abs(diff) < 1.0: return 'steady', change
    return ('rising', change) if diff > 0 else ('falling', change)

def classify_tendency_detailed(slp_series):
    if len(slp_series) < 2: return None, None
    slp_vals = [s for _, s in slp_series if s is not None]
    if len(slp_vals) < 2: return None, None
    first = slp_vals[0]; last = slp_vals[-1]; mid = slp_vals[len(slp_vals)//2]
    diff_total = last - first; diff_first = mid - first; diff_last = last - mid
    STEADY = 1.0
    change = int(round(diff_total * 10))
    def sign(x): return 1 if x > STEADY else (-1 if x < -STEADY else 0)
    s1, s2 = sign(diff_first), sign(diff_last)
    if   s1 ==  1 and s2 ==  1: return 'rising',         change
    elif s1 == -1 and s2 == -1: return 'falling',        change
    elif s1 ==  0 and s2 ==  0: return 'steady',         change
    elif s1 ==  1 and s2 == -1: return 'rising_falling', change
    elif s1 == -1 and s2 ==  1: return 'falling_rising', change
    elif s1 ==  1 and s2 ==  0: return 'rising_steady',  change
    elif s1 == -1 and s2 ==  0: return 'falling_steady', change
    elif s1 ==  0 and s2 ==  1: return 'rising',         change
    elif s1 ==  0 and s2 == -1: return 'falling',        change
    else:                        return 'steady',         change

station_slp_series = defaultdict(list)
for d in metar_records:
    if d['slp'] is not None:
        station_slp_series[d['icao']].append((d['timestamp'], d['slp']))
for icao in station_slp_series:
    station_slp_series[icao].sort(key=lambda x: x[0])

tendency_assigned = 0
for d in metar_records:
    series = [(ts, slp) for ts, slp in station_slp_series[d['icao']]
              if ts <= d['timestamp']]
    if len(series) >= 2:
        tend, change = classify_tendency_detailed(series)
        if tend is not None:
            d['tendency']        = tend
            d['pressure_change'] = change
            tendency_assigned   += 1

for d in metar_records:
    d.setdefault('tendency', None)
    d.setdefault('pressure_change', None)

print(f'✓ Tendency computed for {tendency_assigned} / {len(metar_records)} records')
no_tend = sum(1 for d in metar_records if d['tendency'] is None)
print(f'  No tendency (insufficient history): {no_tend}')

from collections import Counter
tend_counts = Counter(d['tendency'] for d in metar_records if d['tendency'])
for k, v in sorted(tend_counts.items(), key=lambda x: -x[1]):
    print(f'  {k:<20} {v}')

src_counts = Counter(d.get('source','metar') for d in metar_records)
print(f'\n  Source breakdown in metar_records:')
for src, cnt in src_counts.items():
    print(f'    {src:<10} {cnt} records')

# ── Interactive station badge grid ────────────────────────────
_all_stations_5b  = sorted(set(d['icao'] for d in metar_records))
_no_tend_stations = set(d['icao'] for d in metar_records if d['tendency'] is None)
_good_count_5b    = len(_all_stations_5b) - len(_no_tend_stations)

_latest_5b = {}
for d in metar_records:
    if d['icao'] not in _latest_5b or d['timestamp'] > _latest_5b[d['icao']]['timestamp']:
        _latest_5b[d['icao']] = d

def _station_badge_5b(icao):
    d         = _latest_5b.get(icao, {})
    has_gap   = icao in _no_tend_stations
    bg        = '#fff3b0' if has_gap else '#e6faf0'
    bdr       = '#e6a800' if has_gap else '#1a7a3a'
    clr       = '#7a5000' if has_gap else '#145c2c'
    ts        = d.get('timestamp', '—')
    tend      = d.get('tendency', '—') or '—'
    src       = d.get('source', '—')
    gap_str   = '⚠ no tendency (insufficient history)' if has_gap else '✔ tendency computed'
    detail_id = f'tend-detail-{icao}'
    popup_bdr = '#a85c00' if has_gap else '#1a7a3a'
    popup_gap_clr = '#a85c00' if has_gap else '#1a7a3a'
    all_lines = sorted([r for r in metar_records if r['icao'] == icao],
                       key=lambda r: r['timestamp'])
    metar_rows = ''.join(
        f'<tr style="border-bottom:1px solid #eee;">'
        f'<td style="padding:2px 8px;color:#555;white-space:nowrap;">{r["timestamp"]}</td>'
        f'<td style="padding:2px 8px;font-family:monospace;font-size:10px;color:#1a2030;white-space:nowrap;">'
        f'SLP:{r.get("slp","—")} &nbsp; tend:{r.get("tendency","—")}</td>'
        f'</tr>'
        for r in all_lines
    )
    # Build SLP chart data for this station
    slp_points = [(r["timestamp"], r["slp"]) for r in all_lines if r.get("slp") is not None]
    chart_labels = [p[0] for p in slp_points]
    chart_values = [p[1] for p in slp_points]
    chart_id = f'slp-chart-{icao}'
    chart_labels_js = str(chart_labels).replace("'", '"')
    chart_values_js = str(chart_values)
    return (
        f'<span style="display:inline-block;position:relative;margin:2px;">'
        f'<span onclick="'
        f'var p=document.getElementById(\'{detail_id}\');'
        f'document.querySelectorAll(\'.tend-detail-popup\').forEach(function(x){{if(x.id!==\'{detail_id}\')x.style.display=\'none\';}});'
        f'p.style.display=p.style.display===\'none\'?\'block\':\'none\';" '
        f'style="font-family:monospace;font-size:11px;color:{clr};cursor:pointer;'
        f'background:{bg};border:1px solid {bdr};border-radius:3px;'
        f'padding:1px 6px;display:inline-block;">{icao}</span>'
        f'<div id="{detail_id}" class="tend-detail-popup" '
        f'style="display:none;position:absolute;top:20px;left:0;z-index:9999;'
        f'background:#fff;border:2px solid {popup_bdr};border-radius:8px;padding:12px 16px;'
        f'font-family:monospace;font-size:12px;color:#1a2030;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.25);min-width:320px;max-width:600px;">'
        f'<b style="font-size:13px;">{icao}</b> '
        f'<span style="color:#888;font-size:10px;">{d.get("name","")}</span> '
        f'<span style="color:#888;font-size:10px;">· {src} · {len(all_lines)} obs</span>'
        f'<hr style="margin:4px 0;border:none;border-top:1px solid #ccc;">'
        f'<span style="color:{popup_gap_clr};font-size:10px;">{gap_str}</span>'
        f' &nbsp; <span style="font-size:10px;">latest: {ts} &nbsp; tend: {tend}</span>'
        f'<hr style="margin:4px 0;border:none;border-top:1px solid #eee;">'
        f'<canvas id="{chart_id}" width="460" height="160" '
        f'style="width:100%;max-width:460px;height:160px;margin:8px 0;display:block;"></canvas>'
        f'<script>'
        f'(function(){{'
        f'  var labels = {chart_labels_js};'
        f'  var values = {chart_values_js};'
        f'  var ctx = document.getElementById("{chart_id}");'
        f'  if (!ctx) return;'
        f'  var mn = Math.min.apply(null,values)-1, mx = Math.max.apply(null,values)+1;'
        f'  new Chart(ctx, {{'
        f'    type:"line",'
        f'    data:{{'
        f'      labels:labels,'
        f'      datasets:[{{'
        f'        label:"SLP (hPa)",'
        f'        data:values,'
        f'        borderColor:"#1a4a8a",'
        f'        backgroundColor:"rgba(26,74,138,0.08)",'
        f'        pointBackgroundColor:"#1a4a8a",'
        f'        pointRadius:4,'
        f'        borderWidth:2,'
        f'        tension:0.3,'
        f'        fill:true'
        f'      }}]'
        f'    }},'
        f'    options:{{'
        f'      responsive:false,'
        f'      plugins:{{legend:{{display:false}},'
        f'        tooltip:{{callbacks:{{label:function(c){{return c.parsed.y.toFixed(1)+" hPa";}}}}}}}},'
        f'      scales:{{'
        f'        x:{{ticks:{{font:{{size:9}},maxRotation:45}}}},'
        f'        y:{{min:mn,max:mx,ticks:{{font:{{size:9}}}},title:{{display:true,text:"hPa",font:{{size:9}}}}}}'
        f'      }}'
        f'    }}'
        f'  }});'
        f'}})()'
        f'</script>'
        f'<div style="max-height:160px;overflow-y:auto;">'
        f'<table style="border-collapse:collapse;width:100%;font-size:10px;">'
        f'<tr style="background:#f0f4f8;">'
        f'<th style="padding:2px 8px;text-align:left;">Time</th>'
        f'<th style="padding:2px 8px;text-align:left;">SLP / Tendency</th></tr>'
        f'{metar_rows}</table></div>'
        f'<button onclick="document.getElementById(\'{detail_id}\').style.display=\'none\';event.stopPropagation();" '
        f'style="margin-top:8px;font-size:10px;padding:2px 10px;cursor:pointer;'
        f'border:1px solid #aaa;border-radius:3px;background:#f0f0f0;">✕ close</button>'
        f'</div></span>'
    )

_badge_rows_5b = ''.join(_station_badge_5b(icao) for icao in _all_stations_5b)
_all_ts_5b     = sorted(set(d['timestamp'] for d in metar_records))
_slp_5b        = sum(1 for d in metar_records if d['slp'])
_wind_5b       = sum(1 for d in metar_records if d['wind_dir'] is not None)
_temp_5b       = sum(1 for d in metar_records if d['temp'] is not None)


# ── Cell UA-1 . Upper-air station list ────────────────────────────────────
print('--- Upper air station list and location ---')
UPPER_AIR_STATIONS = [
    # ── CANADA ──
    {'id':'CYLT', 'name':'Alert',                 'lat':82.50, 'lon':-62.33,  'wmo':'71082'},
    {'id':'CYEU', 'name':'Eureka',                'lat':79.98, 'lon':-85.93,  'wmo':'71917'},
    {'id':'YUX', 'name':'HALL BEACH ',             'lat':68.77, 'lon':-81.23,  'wmo':'71081'},
    {'id':'YRB',  'name':'Resolute Bay',           'lat':74.72, 'lon':-94.98,  'wmo':'71924'},
    {'id':'YCB',  'name':'Clyde River',            'lat':70.49, 'lon':-68.52,  'wmo':'71925'},
    {'id':'CYFB', 'name':'Iqaluit',                'lat':63.75, 'lon':-68.53,  'wmo':'71909'},
    {'id':'YBK',  'name':'Baker Lake',             'lat':64.30, 'lon':-96.00,  'wmo':'71926'},
    {'id':'CYRK', 'name':'Rankin Inlet',           'lat':62.82, 'lon':-92.12,  'wmo':'71907'},
    {'id':'CYVP', 'name':'Kuujjuaq',               'lat':58.10, 'lon':-68.42,  'wmo':'71906'},
 #   {'id':'CYCO', 'name':'Coral Harbour',          'lat':64.19, 'lon':-83.36,  'wmo':'71815'},
    {'id':'YYR',  'name':'Goose Bay',              'lat':53.31, 'lon':-60.36,  'wmo':'71816'},
 #   {'id':'CYYT', 'name':"St. John's",             'lat':47.62, 'lon':-52.74,  'wmo':'71801'},
 #   {'id':'CYSJ', 'name':'Sable Island',           'lat':43.93, 'lon':-60.02,  'wmo':'71600'},
    {'id':'CYHZ', 'name':'Yarmouth',               'lat':43.83, 'lon':-66.08,  'wmo':'71603'},
    {'id':'CYBG', 'name':'Bagotville',             'lat':48.33, 'lon':-70.99,  'wmo':'71722'},
#    {'id':'CYYY', 'name':'Mont Joli',              'lat':48.60, 'lon':-68.22,  'wmo':'71714'},
#    {'id':'CYBR', 'name':'Brandon',                'lat':49.91, 'lon':-99.95,  'wmo':'71869'},
    {'id':'CYMX', 'name':'Maniwaki',               'lat':46.37, 'lon':-75.98,  'wmo':'71722'},
 #   {'id':'CYLA', 'name':'La Grande IV',           'lat':53.75, 'lon':-73.67,  'wmo':'71823'},
 #   {'id':'CYMO', 'name':'Moosonee',               'lat':51.29, 'lon':-80.60,  'wmo':'71836'},
    {'id':'CYTL', 'name':'Big Trout Lake',         'lat':53.82, 'lon':-89.87,  'wmo':'71845'},
 #   {'id':'CYYU', 'name':'Kapuskasing',            'lat':49.41, 'lon':-82.47,  'wmo':'71731'},
    {'id':'YYQ',  'name':'Churchill',              'lat':58.75, 'lon':-94.07,  'wmo':'71913'},
    {'id':'YQD',  'name':'The Pas',                'lat':53.97, 'lon':-101.10, 'wmo':'71867'},
#    {'id':'CYXE', 'name':'Saskatoon',              'lat':52.17, 'lon':-106.68, 'wmo':'71866'},
    {'id':'WSE', 'name':'Edmonton Stony Plain',   'lat':53.55, 'lon':-114.11, 'wmo':'71119'},
    {'id':'YSM',  'name':'Fort Smith',             'lat':60.02, 'lon':-111.96, 'wmo':'71934'},
    {'id':'YVQ',  'name':'Norman Wells',           'lat':65.28, 'lon':-126.80, 'wmo':'71043'},
    {'id':'YXY',  'name':'Whitehorse',             'lat':60.72, 'lon':-135.07, 'wmo':'71964'},
    {'id':'YYE',  'name':'Fort Nelson',            'lat':58.84, 'lon':-122.60, 'wmo':'71945'},
    {'id':'YEV',  'name':'Inuvik',                 'lat':68.30, 'lon':-133.48, 'wmo':'71957'},
    {'id':'ZXS', 'name':'Prince George',          'lat':53.88, 'lon':-122.68, 'wmo':'71908'},
    {'id':'WLW', 'name':'Vernon',                 'lat':50.24, 'lon':-119.29, 'wmo':'73033'},
    {'id':'YZT',  'name':'Port Hardy',             'lat':50.68, 'lon':-127.37, 'wmo':'71109'},
    # ── ALASKA ──
    #{'id':'BRW',  'name':'Utqiagvik (Barrow)',     'lat':71.30, 'lon':-156.78, 'wmo':'70026'},
  #  {'id':'OTZ',  'name':'Kotzebue',               'lat':66.87, 'lon':-162.63, 'wmo':'70133'},
    {'id':'OME',  'name':'Nome',                   'lat':64.50, 'lon':-165.43, 'wmo':'70200'},
    {'id':'BET',  'name':'Bethel',                 'lat':60.78, 'lon':-161.80, 'wmo':'70219'},
    {'id':'MCG',  'name':'McGrath',                'lat':62.97, 'lon':-155.62, 'wmo':'70231'},
    {'id':'FAI',  'name':'Fairbanks',              'lat':64.82, 'lon':-147.87, 'wmo':'70261'},
    {'id':'ANC',  'name':'Anchorage',              'lat':61.17, 'lon':-150.02, 'wmo':'70273'},
    {'id':'AKN',  'name':'King Salmon',            'lat':58.68, 'lon':-156.65, 'wmo':'70326'},
    {'id':'PACD', 'name':'Cold Bay',               'lat':55.20, 'lon':-162.73, 'wmo':'70316'},
    {'id':'PADQ', 'name':'Kodiak',                 'lat':57.75, 'lon':-152.50, 'wmo':'70350'},
   # {'id':'NHB',  'name':'St Paul Island',         'lat':57.15, 'lon':-170.22, 'wmo':'70308'},
    {'id':'ANN',  'name':'Annette Island',         'lat':55.03, 'lon':-131.57, 'wmo':'70398'},
    {'id':'YAK',  'name':'Yakutat',                'lat':59.52, 'lon':-139.67, 'wmo':'70361'},
    # ── CONTIGUOUS US ──
    {'id':'OAK',  'name':'Salem',                  'lat':44.91, 'lon':-123.01, 'wmo':'72694'},
    {'id':'UIL',  'name':'Quillayute WA',          'lat':47.95, 'lon':-124.55, 'wmo':'72797'},
    {'id':'SLC',  'name':'Salt Lake City UT',      'lat':40.77, 'lon':-111.97, 'wmo':'72572'},
    {'id':'BOI',  'name':'Boise ID',               'lat':43.57, 'lon':-116.22, 'wmo':'72681'},
    {'id':'EKO',  'name':'Elko NV',                'lat':40.87, 'lon':-115.73, 'wmo':'72582'},
    {'id':'KOAK', 'name':'Oakland CA',             'lat':37.73, 'lon':-122.22, 'wmo':'72493'},
  #  {'id':'KVBG', 'name':'Vandenberg CA',          'lat':34.73, 'lon':-120.57, 'wmo':'72393'},
    {'id':'MFR',  'name':'Medford OR',             'lat':42.37, 'lon':-122.87, 'wmo':'72597'},
    {'id':'GGW',  'name':'Glasgow MT',             'lat':48.21, 'lon':-106.62, 'wmo':'72768'},
    {'id':'RIW',  'name':'Riverton WY',            'lat':43.07, 'lon':-108.48, 'wmo':'72672'},
    {'id':'KGJT', 'name':'Grand Junction CO',      'lat':39.12, 'lon':-108.53, 'wmo':'72476'},
    {'id':'KFGZ', 'name':'Flagstaff AZ',           'lat':35.23, 'lon':-111.82, 'wmo':'72376'},
    {'id':'KVEF', 'name':'Las Vegas NV',           'lat':36.08, 'lon':-115.17, 'wmo':'72388'},
    {'id':'KABQ', 'name':'Albuquerque NM',         'lat':35.05, 'lon':-106.62, 'wmo':'72365'},
    {'id':'BIS',  'name':'Bismarck ND',            'lat':46.77, 'lon':-100.75, 'wmo':'72764'},
    {'id':'ABR',  'name':'Aberdeen SD',            'lat':45.45, 'lon':-98.42,  'wmo':'72659'},
    {'id':'RAP',  'name':'Rapid City SD',          'lat':44.07, 'lon':-103.22, 'wmo':'72662'},
    {'id':'KDDC', 'name':'Dodge City KS',          'lat':37.77, 'lon':-99.97,  'wmo':'72451'},
    {'id':'KTOP', 'name':'Topeka KS',              'lat':39.07, 'lon':-95.63,  'wmo':'72456'},
    {'id':'KAMA', 'name':'Amarillo TX',            'lat':35.22, 'lon':-101.72, 'wmo':'72363'},
    {'id':'OUN',  'name':'Norman OK',              'lat':35.18, 'lon':-97.43,  'wmo':'72357'},
 #   {'id':'KEPZ', 'name':'Santa Teresa NM',        'lat':31.87, 'lon':-106.70, 'wmo':'72270'},
    {'id':'KMAF', 'name':'Midland TX',             'lat':31.95, 'lon':-102.18, 'wmo':'72265'},
    {'id':'FWD', 'name':'Fort Worth TX',             'lat':32.84, 'lon':-97.30, 'wmo':'72249'},
    {'id':'KDRT', 'name':'Del Rio TX',             'lat':29.37, 'lon':-100.92, 'wmo':'72261'},
    {'id':'KCRP', 'name':'Corpus Christi TX',      'lat':27.77, 'lon':-97.50,  'wmo':'72251'},
    {'id':'INL',  'name':'International Falls MN', 'lat':48.57, 'lon':-93.38,  'wmo':'72747'},
    {'id':'MPX',  'name':'Minneapolis MN',         'lat':44.85, 'lon':-93.57,  'wmo':'72649'},
    {'id':'LBF',  'name':'North Platte NE',        'lat':41.13, 'lon':-100.68, 'wmo':'72562'},
  ##  {'id':'OAX',  'name':'Omaha NE',               'lat':41.32, 'lon':-96.37,  'wmo':'72553'},
    {'id':'KILX', 'name':'Lincoln IL',             'lat':40.15, 'lon':-89.33,  'wmo':'72440'},
    {'id':'KDVN', 'name':'Davenport IA',           'lat':41.62, 'lon':-90.58,  'wmo':'74455'},
    {'id':'KSGF', 'name':'Springfield MO',         'lat':37.23, 'lon':-93.38,  'wmo':'72440'},
    {'id':'KBUF', 'name':'Buffalo NY',             'lat':42.93, 'lon':-78.73,  'wmo':'72528'},
    {'id':'KGYX', 'name':'Gray ME',                'lat':43.90, 'lon':-70.25,  'wmo':'74389'},
    {'id':'KSHV', 'name':'Shreveport LA',          'lat':32.45, 'lon':-93.82,  'wmo':'72248'},
    {'id':'KLCH', 'name':'Lake Charles LA',        'lat':30.12, 'lon':-93.22,  'wmo':'72240'},
    {'id':'KLIX', 'name':'New Orleans LA',         'lat':30.33, 'lon':-89.82,  'wmo':'72233'},
    {'id':'PHTO', 'name':'Hilo',                   'lat':19.72, 'lon':-155.05, 'wmo':'91285'},
    {'id':'PHLI', 'name':'Lihue',                  'lat':21.99, 'lon':-159.34, 'wmo':'91165'},

]

print(f'✓ {len(UPPER_AIR_STATIONS)} upper-air stations defined')

# ── Cell UA-2 . Fetch Wyoming soundings and display raw data table ─────────

import re
import concurrent.futures
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
import requests
import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────
WYOMING_BASE = 'https://weather.uwyo.edu/wsgi/sounding'
UA_HOURS     = [0, 12]
UA_WORKERS   = 12
UA_TIMEOUT   = 12





# ====================================================
#=====================================================
import threading, time

##########################################################





# ── Step 1: build datetime string for latest 00Z / 12Z ────────────────────
def get_sounding_dt(hour):
    now = datetime.now(timezone.utc)
    dt  = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if dt > now:
        dt -= timedelta(days=1)
    return dt


# ── Step 2: fetch raw HTML for one station/hour ────────────────────────────
def fetch_raw(stn, hour):
    dt     = get_sounding_dt(hour)
    dt_str = f'{dt.strftime("%Y-%m-%d")} {hour}:00:00'
    url    = f'{WYOMING_BASE}?datetime={quote(dt_str)}&id={stn["wmo"]}&src=BUFR&type=TEXT:LIST'
    try:
        r = requests.get(url, timeout=UA_TIMEOUT)
        if not r.ok or len(r.text) < 300:
            return None
        txt = r.text
        if 'Can\'t get' in txt or 'No Observation' in txt or 'ERROR' in txt:
            return None
        return txt
    except Exception:
        return None


# ── Step 3: parse raw HTML → list of level dicts ──────────────────────────
def parse_sounding(html, stn, hour):
    dt = get_sounding_dt(hour)

    # station header
    obs_name = obs_lat = obs_lon = None
    lat_m = re.search(r'Latitude:\s*([-\d.]+)', html, re.I)
    lon_m = re.search(r'Longitude:\s*([-\d.]+)', html, re.I)
    if lat_m: obs_lat = float(lat_m.group(1))
    if lon_m: obs_lon = float(lon_m.group(1))
    h2 = re.search(r'<h2[^>]*>([\s\S]*?)</h2>', html, re.I)
    if h2:
        txt = re.sub(r'<[^>]+>', ' ', h2.group(1))
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        if len(lines) >= 2:
            obs_name = lines[-1]

    # data table
    pre = re.search(r'<pre>([\s\S]*?)</pre>', html, re.I)
    if not pre:
        return []

    rows = []
    for line in pre.group(1).split('\n'):
        cols = line.strip().split()
        if len(cols) < 7:
            continue
        try:
            pres = float(cols[0])
        except ValueError:
            continue
        if not (100 <= pres <= 1100):
            continue

        def fv(i):
            try:
                v = float(cols[i])
                return None if abs(v) > 9000 else v
            except (IndexError, ValueError):
                return None

        rows.append({
            'icao':       stn['id'],
            'wmo':        stn['wmo'],
            'stn_name':   obs_name or stn['name'],
            'lat':        obs_lat  or stn['lat'],
            'lon':        obs_lon  or stn['lon'],
            'valid_time': dt.strftime('%Y-%m-%d') + f' {hour:02d}Z',
            'hour':       hour,
            'PRES':       pres,
            'HGHT':       fv(1),
            'TEMP':       fv(2),
            'DWPT':       fv(3),
            'RELH':       fv(4),
            'MIXR':       fv(5),
            'DRCT':       fv(6),
            'SPED':       fv(7),
            'THTA':       fv(8)  if len(cols) > 8  else None,
            'THTE':       fv(9)  if len(cols) > 9  else None,
            'THTV':       fv(10) if len(cols) > 10 else None,
        })
    return rows


# ── Step 4: fetch all stations with progress bar ───────────────────────────
def fetch_all(stations, hours=UA_HOURS, workers=UA_WORKERS):
    tasks   = [(s, h) for s in stations for h in hours]
    total   = len(tasks)
    done    = 0
    ok      = 0
    failed  = []
    all_rows = []

    bar_id  = 'ua2-bar'
    stat_id = 'ua2-stat'
    log_id  = 'ua2-log'


    def _ui(msg, color='#333'):
        pct  = int(done / total * 100) if total else 0
        safe = msg.replace('\\', '\\\\').replace("'", "\\'").replace('\n', ' ')

    # Pass 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fmap = {ex.submit(fetch_raw, s, h): (s, h) for s, h in tasks}
        for fut in concurrent.futures.as_completed(fmap):
            s, h = fmap[fut]
            done += 1
            html = fut.result()
            if html:
                rows = parse_sounding(html, s, h)
                if rows:
                    ok += 1
                    all_rows.extend(rows)
                    _ui(f'✔ {s["id"]:6s} {h:02d}Z  {len(rows)} levels', '#1a5c1a')
                else:
                    failed.append((s, h))
                    _ui(f'✘ {s["id"]:6s} {h:02d}Z  parse failed', '#cc4400')
            else:
                failed.append((s, h))
                _ui(f'✘ {s["id"]:6s} {h:02d}Z  no data', '#cc4400')

    # Pass 2: retry with 1-day lookback
    if failed:
        _ui(f'↻ Retrying {len(failed)} with 1-day lookback...', '#aa6600')
        retry = failed[:]
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fmap2 = {ex.submit(fetch_raw, s, h): (s, h) for s, h in retry}
            for fut in concurrent.futures.as_completed(fmap2):
                s, h = fmap2[fut]
                html = fut.result()
                if html:
                    rows = parse_sounding(html, s, h)
                    if rows:
                        ok += 1
                        all_rows.extend(rows)
                        _ui(f'↻✔ {s["id"]:6s} {h:02d}Z  recovered ({len(rows)} levels)', '#2266aa')
                        continue
                failed.append((s, h))
                _ui(f'↻✘ {s["id"]:6s} {h:02d}Z  no data after retry', '#aa0000')

    # ── Build status report ──────────────────────────────────────────────
    # track per station/hour outcome
    _status = {}  # (icao, hour) -> '✔ ok' | '↻✔ recovered' | '✘ no data'
    _retry_set  = {(s['id'], h) for s, h in retry} if 'retry' in locals() else set()
    _failed_set = {(s['id'], h) for s, h in failed}
    _recovered_keys = _retry_set - _failed_set
    for _r in all_rows:
        _k = (_r['icao'], _r['hour'])
        if _k not in _status:
            _status[_k] = '↻✔ recovered' if _k in _recovered_keys else '✔ ok'
    for _s, _h in failed:
        _status[(_s['id'], _h)] = '✘ no data'

    # collect all station ids in original order
    _all_ids = []
    _seen = set()
    for _s, _h in tasks:
        if _s['id'] not in _seen:
            _all_ids.append(_s['id'])
            _seen.add(_s['id'])

# Build a dict: (icao, hour) -> valid_time string (from parsed rows)
    _valid_time_map = {}
    for _r in all_rows:
        _k = (_r['icao'], _r['hour'])
        _valid_time_map[_k] = _r['valid_time']  # last write wins, all same per station/hour

    # Canonical date per hour (majority vote, same logic as filtering block)
    _canonical_date = {}
    for _h in hours:
        _vts = [_valid_time_map[_k] for _k in _valid_time_map if _k[1] == _h]
        if _vts:
            from collections import Counter
            _canonical_date[_h] = Counter(_vts).most_common(1)[0][0]

    def _cell(icao, hour):
        v  = _status.get((icao, hour))
        vt = _valid_time_map.get((icao, hour))
        canonical = _canonical_date.get(hour)
        is_stale  = vt and canonical and (vt != canonical)

        date_badge = ''
        if vt:
            # show just the date portion (first 10 chars of 'YYYY-MM-DD HHZ')
            date_str = vt[:10]
            if is_stale:
                date_badge = (f'<br><span style="font-size:9px;background:#ff9900;color:#fff;'
                              f'border-radius:2px;padding:0 3px">⚠ {date_str}</span>')
            else:
                date_badge = (f'<br><span style="font-size:9px;color:#666">{date_str}</span>')

        if v == '✔ ok' and not is_stale:
            return f'<td style="background:#c8f0d0;color:#1a5c1a;text-align:center;font-weight:bold">✔{date_badge}</td>'
        elif v == '✔ ok' and is_stale:
            return f'<td style="background:#ffe8b0;color:#7a4400;text-align:center;font-weight:bold">⚠{date_badge}</td>'
        elif v == '↻✔ recovered' and not is_stale:
            return f'<td style="background:#cce0ff;color:#1a3a8a;text-align:center;font-weight:bold">↻✔{date_badge}</td>'
        elif v == '↻✔ recovered' and is_stale:
            return f'<td style="background:#ffe8b0;color:#7a4400;text-align:center;font-weight:bold">↻⚠{date_badge}</td>'
        elif v == '✘ no data':
            return '<td style="background:#ffd6cc;color:#aa0000;text-align:center;font-weight:bold">✘</td>'
        else:
            return '<td style="background:#f0f0f0;color:#aaa;text-align:center">—</td>'

    _rows_html = ''
    for _id in _all_ids:
        _rows_html += f'<tr><td style="font-family:Courier New,monospace;font-size:11px;padding:2px 8px">{_id}</td>'
        for _h in hours:
            _rows_html += _cell(_id, _h)
        _rows_html += '</tr>\n'

    _legend = (
        '<span style="background:#c8f0d0;padding:2px 8px;border-radius:3px;margin-right:6px">✔ ok</span>'
        '<span style="background:#cce0ff;padding:2px 8px;border-radius:3px;margin-right:6px">↻✔ recovered</span>'
        '<span style="background:#ffe8b0;padding:2px 8px;border-radius:3px;margin-right:6px">⚠ stale date</span>'
        '<span style="background:#ffd6cc;padding:2px 8px;border-radius:3px">✘ no data</span>'
    )
    _hrs_header = ''.join(f'<th style="padding:2px 12px">{h:02d}Z</th>' for h in hours)

    col = '#1a7a3a' if not failed else '#cc4400'
    bg  = '#b6f5c8' if not failed else '#ffe8d6'
    return all_rows, _status


# ── Run ────────────────────────────────────────────────────────────────────
raw_rows, _status = fetch_all(UPPER_AIR_STATIONS)

# ── Build raw data table ───────────────────────────────────────────────────
print(f'Parsing {len(raw_rows)} level rows into DataFrame...', end=' ', flush=True)
ua_raw_df = pd.DataFrame(raw_rows, columns=[
    'icao','wmo','stn_name','lat','lon','valid_time','hour',
    'PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV'
])
# ── Determine canonical valid_time per hour (majority vote) ───────────────
def _canonical_time(group):
    return group['valid_time'].value_counts().index[0]

_canonical = ua_raw_df.groupby('hour').apply(_canonical_time).rename('canonical_time')
ua_raw_df = ua_raw_df.merge(_canonical, on='hour')

# Flag stale rows (valid_time doesn't match the canonical time for that hour)
ua_raw_df['is_stale'] = ua_raw_df['valid_time'] != ua_raw_df['canonical_time']

# Keep recovered (lookback=1) rows only for stations with no fresh data at all
_fresh      = ua_raw_df[~ua_raw_df['is_stale']].copy()
_stale      = ua_raw_df[ ua_raw_df['is_stale']].copy()
_fresh_keys = set(zip(_fresh['icao'], _fresh['hour']))
_stale_keys = set(zip(_stale['icao'], _stale['hour']))
_keep_stale = _stale[
    pd.MultiIndex.from_arrays([_stale['icao'], _stale['hour']]).isin(_stale_keys - _fresh_keys)
].copy()

ua_raw_df = pd.concat([_fresh, _keep_stale], ignore_index=True)
ua_raw_df = ua_raw_df.drop(columns=['canonical_time'])

_stale_stns = ua_raw_df[ua_raw_df['is_stale']]['icao'].unique().tolist()
print(f'  → {(~ua_raw_df["is_stale"]).sum()} fresh rows, '
      f'{ua_raw_df["is_stale"].sum()} stale rows dropped from stations: {_stale_stns or "none"}')
ua_raw_df = ua_raw_df[~ua_raw_df['is_stale']].drop(columns=['is_stale']).reset_index(drop=True)
print('done.')

# ── Show valid times ───────────────────────────────────────────────────────
_vt_summary = (
    ua_raw_df.groupby(['hour','valid_time'])
    .agg(stations=('icao','nunique'), levels=('PRES','count'))
    .reset_index()
    .sort_values(['hour','valid_time'])
)
print(f'\n✓ {len(ua_raw_df)} rows  |  '
      f'{ua_raw_df["icao"].nunique()} stations  |  '
      f'{ua_raw_df["valid_time"].nunique()} valid times\n')
print(f'{"Hour":>6}  {"Valid Time":<20}  {"Stations":>8}  {"Levels":>8}')
print('-' * 50)
for _, _vr in _vt_summary.iterrows():
    print(f'{int(_vr["hour"]):>4}Z  {str(_vr["valid_time"]):<20}  '
          f'{int(_vr["stations"]):>8}  {int(_vr["levels"]):>8}')

# ── Display ────────────────────────────────────────────────────────────────
_uid = 'uaraw'
_N   = 10

def _style_raw(df):
    s = df.style.format(na_rep='—', precision=1)
    for c, cm in [('TEMP','RdBu_r'),('DWPT','BuGn'),
                  ('SPED','Purples'),('RELH','Blues'),('HGHT','YlOrRd')]:
        if c in df.columns:
            s = s.background_gradient(subset=[c], cmap=cm)
    return s

_sh = _style_raw(ua_raw_df.head(_N)).to_html()
_fl = _style_raw(ua_raw_df.head(500)).to_html()  # cap at 500 rows for display








#####################################################

import folium
import pandas as pd
import numpy as np


print('Upper Air stations data availability')

df = ua_raw_df.copy()


# ─────────────────────────────────────────────
# helper: safe formatting
# ─────────────────────────────────────────────
def fmt(val, spec=".1f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return format(val, spec)


# ─────────────────────────────────────────────
# station stats per timestep
# ─────────────────────────────────────────────
def stats(sub):
    if sub is None or len(sub) == 0:
        return None

    return {
        "levels": len(sub),
        "temp_mean": sub["TEMP"].mean(),
        "wind_mean": sub["SPED"].mean(),
        "min_pres": sub["PRES"].min(),
        "max_pres": sub["PRES"].max(),
    }


# ─────────────────────────────────────────────
# status scoring
# ─────────────────────────────────────────────
def score(st):
    return {
        "✔ ok": 2,
        "↻✔ recovered": 1,
        "✘ no data": 0
    }.get(st, 0)


# ─────────────────────────────────────────────
# map
# ─────────────────────────────────────────────
m = folium.Map(location=[55, -100], zoom_start=3)


for s in UPPER_AIR_STATIONS:
    sid = s["id"]

    # split timesteps
    sub00 = df[(df["icao"] == sid) & (df["hour"] == 0)]
    sub12 = df[(df["icao"] == sid) & (df["hour"] == 12)]

    st00 = _status.get((sid, 0), "✘ no data")
    st12 = _status.get((sid, 12), "✘ no data")

    d00 = stats(sub00)
    d12 = stats(sub12)


    # ─────────────────────────────
    # popup HTML
    # ─────────────────────────────
    popup_html = f"""
<div style="
    font-family: Arial;
    font-size: 12px;
    width: 360px;
">

    <div style="font-weight:bold;margin-bottom:6px;">
        {s['name']} ({sid})
    </div>

    <!-- FLEX CONTAINER -->
    <div style="display:flex; gap:6px;">

        <!-- ───────── LEFT: 00Z ───────── -->
        <div style="
            flex:1;
            border:1px solid #ccc;
            padding:6px;
            border-radius:4px;
            background:#f7fbff;
        ">
            <b>🕛 00Z</b><br>
            Status: {st00}<br>
            Levels: {d00['levels'] if d00 else '—'}<br>
            Temp: {fmt(d00['temp_mean']) if d00 else '—'} °C<br>
            Wind: {fmt(d00['wind_mean']) if d00 else '—'} kt<br>
            P: {fmt(d00['min_pres'], '.0f') if d00 else '—'}–{fmt(d00['max_pres'], '.0f') if d00 else '—'}
        </div>

        <!-- ───────── RIGHT: 12Z ───────── -->
        <div style="
            flex:1;
            border:1px solid #ccc;
            padding:6px;
            border-radius:4px;
            background:#fffaf3;
        ">
            <b>🕕 12Z</b><br>
            Status: {st12}<br>
            Levels: {d12['levels'] if d12 else '—'}<br>
            Temp: {fmt(d12['temp_mean']) if d12 else '—'} °C<br>
            Wind: {fmt(d12['wind_mean']) if d12 else '—'} kt<br>
            P: {fmt(d12['min_pres'], '.0f') if d12 else '—'}–{fmt(d12['max_pres'], '.0f') if d12 else '—'}
        </div>

    </div>
</div>
"""


    # ─────────────────────────────
    # color logic (your rules)
    # ─────────────────────────────
    a = score(st00)
    b = score(st12)

    if a == 2 and b == 2:
        color = "green"

    elif (a >= 1 and b >= 1):
        # both partial/recovered or mixed good/recovered
        color = "blue"   # light blue

    elif (a > 0 and b == 0) or (a == 0 and b > 0):
        color = "orange"

    else:
        color = "red"




# ─────────────────────────────
# INVISIBLE HITBOX (large clickable area)
# ─────────────────────────────
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=15,              # click area ONLY
        color="white",
        fill=True,
        fill_color="white",
        fill_opacity=0.4,
        weight=0,
        popup=folium.Popup(popup_html, max_width=380)
    ).add_to(m)

    # ─────────────────────────────
    # marker
    # ─────────────────────────────
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=380)
    ).add_to(m)


m

# Skipping Upper Air station validation check?
SKIP = True



if not SKIP:
    import signal, json, os
    from datetime import datetime, timezone, timedelta

    class _Skip(Exception): pass
    def _timeout_handler(signum, frame): raise _Skip()

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(10)

        print('--- Upper air station fetching - Successful report ---')

        # ── Mount Drive ────────────────────────────────────────────────────────

        os.makedirs(FAILURE_LOG_DIR, exist_ok=True)
        FAILURE_LOG_PATH = f'{FAILURE_LOG_DIR}/ua_station_failures.json'
        SUGGEST_REMOVE_DAYS = 10

        # ── Load existing log ──────────────────────────────────────────────────
        if os.path.exists(FAILURE_LOG_PATH):
            with open(FAILURE_LOG_PATH) as _f:
                _failure_log = json.load(_f)
            print(f'Loaded failure log: {len(_failure_log)} stations tracked')
        else:
            _failure_log = {}
            print('No existing failure log — starting fresh')

        # ── Today's date string ────────────────────────────────────────────────
        _today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # ── Update log from this run ───────────────────────────────────────────
        _all_station_ids = list({s['id'] for s in UPPER_AIR_STATIONS})

        for _sid in _all_station_ids:
            _h0   = _status.get((_sid, 0),  '')
            _h12  = _status.get((_sid, 12), '')
            _any_ok = ('✔' in _h0) or ('✔' in _h12)
            if _sid not in _failure_log:
                _failure_log[_sid] = {'fail_dates': [], 'ok_dates': []}
            if _any_ok:
                if _today not in _failure_log[_sid]['ok_dates']:
                    _failure_log[_sid]['ok_dates'].append(_today)
            else:
                if _today not in _failure_log[_sid]['fail_dates']:
                    _failure_log[_sid]['fail_dates'].append(_today)

        # ── Save updated log ───────────────────────────────────────────────────
        with open(FAILURE_LOG_PATH, 'w') as _f:
            json.dump(_failure_log, _f, indent=2)
        print(f'Failure log saved → {FAILURE_LOG_PATH}')

        # ── Compute consecutive recent fail days ───────────────────────────────
        def _consec_fail_days(entry):
            fail_set = set(entry.get('fail_dates', []))
            ok_set   = set(entry.get('ok_dates',   []))
            count = 0
            check = datetime.now(timezone.utc).date()
            for _ in range(60):
                ds = check.strftime('%Y-%m-%d')
                if ds in ok_set:   break
                if ds in fail_set: count += 1
                check -= timedelta(days=1)
            return count

        # ── Build report ───────────────────────────────────────────────────────
        _report = []
        for _sid in sorted(_failure_log.keys()):
            _e = _failure_log[_sid]
            _consec = _consec_fail_days(_e)
            _report.append({
                'icao':        _sid,
                'consec_fail': _consec,
                'total_fail':  len(_e.get('fail_dates', [])),
                'total_ok':    len(_e.get('ok_dates',   [])),
                'last_ok':     max(_e['ok_dates']) if _e['ok_dates'] else '—',
                'suggest':     _consec >= SUGGEST_REMOVE_DAYS,
            })
        _report.sort(key=lambda x: -x['consec_fail'])

        # ── Display table ──────────────────────────────────────────────────────
        def _row_html(r):
            bg         = '#fff3cd' if r['suggest'] else '#ffffff'
            flag       = '⚠ Remove?' if r['suggest'] else ''
            flag_col   = '#cc6600' if r['suggest'] else '#888'
            consec_col = '#cc0000' if r['consec_fail'] >= SUGGEST_REMOVE_DAYS else (
                         '#cc6600' if r['consec_fail'] >= 5 else '#333')
            return (
                f'<tr style="background:{bg}">'
                f'<td style="font-family:Courier New,monospace;padding:3px 10px;font-weight:bold">{r["icao"]}</td>'
                f'<td style="text-align:center;color:{consec_col};font-weight:bold;padding:3px 10px">{r["consec_fail"]}</td>'
                f'<td style="text-align:center;padding:3px 10px">{r["total_fail"]}</td>'
                f'<td style="text-align:center;padding:3px 10px">{r["total_ok"]}</td>'
                f'<td style="text-align:center;padding:3px 10px;color:#555">{r["last_ok"]}</td>'
                f'<td style="text-align:center;color:{flag_col};font-weight:bold;padding:3px 10px">{flag}</td>'
                f'</tr>'
            )

        _rows_html    = ''.join(_row_html(r) for r in _report)
        _suggest_list = [r['icao'] for r in _report if r['suggest']]

        _suggest_box = '' if not _suggest_list else f'''
        <div style="margin-top:10px;padding:10px 16px;background:#fff3cd;border:2px solid #cc6600;
                    border-radius:8px;font-family:Courier New,monospace;font-size:12px;">
          <b style="color:#cc6600">⚠ Suggested removals ({len(_suggest_list)} stations):</b><br>
          <span style="color:#333">{", ".join(_suggest_list)}</span><br><br>
          <span style="color:#888;font-size:10px">
            Remove these from UPPER_AIR_STATIONS to speed up fetching.
          </span>
        </div>'''


    except _Skip:
        print("⚠ Station failure tracker: timed out (>10s) — skipping")
    except Exception as _e:
        print(f"⚠ Station failure tracker: skipped ({type(_e).__name__}: {_e})")
    finally:
        signal.alarm(0)

else:
    print("⏭ Skipped — Skipping Upper air station validation check")




# ── Cell UA-2b. Fetch EC model upper-air data and merge into ua_raw_df ─────────

import math as _math
import requests
import pandas as pd
from datetime import datetime, timezone as _tz

# ── Config ──────────────────────────────────────────────────────────────────
OPENMETEO_PRESSURE_URL = 'https://api.open-meteo.com/v1/forecast'

# Standard pressure levels available from Open-Meteo ECMWF IFS pressure-level API
EC_PRESSURE_LEVELS = [850, 700, 500, 250]

# Virtual station grid — same transect as surface EC stations
EC_UA_LONGITUDE = -139.7                   # 139.7°W — PAYA (Yakutat) meridian
EC_UA_LATITUDES = list(range(49, 0, -10))  # [49, 39, 29, 19, 9]

EC_UA_PAST_DAYS     = 1
EC_UA_FORECAST_DAYS = 2


# ── Helper: build ICAO-style ID (reuse existing ec_icao if available) ────────
def ec_ua_icao(lat, lon):
    return f"ECML{abs(lat):02d}"


# ── Fetch pressure-level variables for one grid point ───────────────────────
def _fetch_ec_ua(lat, lon):
    """
    Returns Open-Meteo JSON with hourly pressure-level fields for ECMWF IFS.
    Variable naming convention: <var>_<hPa>hPa
    """
    # Build variable list: temperature, geopotential, wind u/v, relative humidity,
    # dewpoint is derived (RH + T → Td)
    vars_per_level = [
        'temperature_{p}hPa',
        'geopotential_height_{p}hPa',
        'wind_speed_{p}hPa',
        'wind_direction_{p}hPa',
        'relative_humidity_{p}hPa',
    ]
    hourly_vars = ','.join(
        tmpl.format(p=p)
        for p in EC_PRESSURE_LEVELS
        for tmpl in vars_per_level
    )

    r = requests.get(OPENMETEO_PRESSURE_URL, params={
        'latitude':        lat,
        'longitude':       lon,
        'hourly':          hourly_vars,
        'models':          'best_match',
        'past_days':       2,
        'forecast_days':   1,
        'timezone':        'UTC',
    }, timeout=25)
    r.raise_for_status()
    return r.json()


# ── Dewpoint from RH + T (Magnus formula) ────────────────────────────────────
def _rh_to_dwpt(temp_c, rh_pct):
    if temp_c is None or rh_pct is None or rh_pct <= 0:
        return None
    a, b = 17.625, 243.04
    alpha = _math.log(max(rh_pct, 0.1) / 100.0) + (a * temp_c / (b + temp_c))
    return round(b * alpha / (a - alpha), 2)


# ── Mixing ratio from dewpoint + pressure (g/kg) ─────────────────────────────
def _dwpt_to_mixr(dwpt_c, pres_hpa):
    if dwpt_c is None or pres_hpa is None:
        return None
    e  = 6.112 * _math.exp(17.67 * dwpt_c / (dwpt_c + 243.5))  # hPa
    w  = 621.97 * e / (pres_hpa - e)                            # g/kg
    return round(w, 3)


# ── Potential temperature (theta) from T + P ─────────────────────────────────
def _theta(temp_c, pres_hpa):
    if temp_c is None or pres_hpa is None:
        return None
    tk = temp_c + 273.15
    return round(tk * (1000.0 / pres_hpa) ** 0.2854, 2)


# ── Equivalent potential temperature (theta-e, Bolton 1980) ──────────────────
def _theta_e(temp_c, dwpt_c, pres_hpa):
    if any(v is None for v in [temp_c, dwpt_c, pres_hpa]):
        return None
    tk  = temp_c + 273.15
    td  = dwpt_c + 273.15
    e   = 6.112 * _math.exp(17.67 * dwpt_c / (dwpt_c + 243.5))
    r   = 0.622 * e / (pres_hpa - e)           # mixing ratio kg/kg
    tlc = 1.0 / (1.0 / (td - 56.0) + _math.log(tk / td) / 800.0) + 56.0
    return round(tk * (1000.0 / pres_hpa) ** (0.2854 * (1 - 0.28e-3 * r * 1000))
                 * _math.exp((3376.0 / tlc - 2.54) * r * (1 + 0.81e-3 * r * 1000)), 2)


# ── Virtual potential temperature (theta-v) ───────────────────────────────────
def _theta_v(temp_c, mixr_gkg, pres_hpa):
    if any(v is None for v in [temp_c, mixr_gkg, pres_hpa]):
        return None
    tk = temp_c + 273.15
    r  = mixr_gkg / 1000.0
    th = tk * (1000.0 / pres_hpa) ** 0.2854
    return round(th * (1 + 1.608 * r) / (1 + r), 2)


# ── Parse one Open-Meteo response → list of ua_raw_df-schema rows ─────────────
def _parse_ec_ua(lat, lon, data):
    icao   = ec_ua_icao(lat, lon)
    hourly = data.get('hourly', {})
    times  = hourly.get('time', [])
    rows   = []

    def col(k):
        return hourly.get(k, [])

    # ── Find single latest timestamp per synoptic hour ────────────────────
    _best = {}
    for i, iso in enumerate(times):
        try:
            dt = datetime.fromisoformat(iso).replace(tzinfo=_tz.utc)
        except Exception:
            continue
        if dt.hour in UA_HOURS:
            if dt.hour not in _best or dt > _best[dt.hour][0]:
                _best[dt.hour] = (dt, i)

    print(f'  [{icao}] best indices: { {h: (str(dt), i) for h,(dt,i) in _best.items()} }')

    _best_indices = {i: (dt, h) for h, (dt, i) in _best.items()}

    for i, iso in enumerate(times):
        if i not in _best_indices:
            continue
        dt, hour   = _best_indices[i]
        valid_time = dt.strftime('%Y-%m-%d') + f' {dt.hour:02d}Z'

        for pres in EC_PRESSURE_LEVELS:
            def gv(tmpl, _p=pres, _i=i):
                key = tmpl.format(p=_p)
                lst = col(key)
                v   = lst[_i] if _i < len(lst) else None
                return None if (v is None or abs(v) > 99000) else v

            temp = gv('temperature_{p}hPa')
            hght = gv('geopotential_height_{p}hPa')
            rh   = gv('relative_humidity_{p}hPa')
            wspd_kmh = gv('wind_speed_{p}hPa')
            wspd = round(wspd_kmh / 1.852, 1) if wspd_kmh is not None else None  # km/h → kt
            wdir = gv('wind_direction_{p}hPa')

            # ── Debug first station first level ──────────────────────────
            if lat == EC_UA_LATITUDES[0] and pres == 850:
                print(f'  [{icao}] 850hPa i={i} temp={temp} hght={hght} rh={rh} wspd={wspd}kt wdir={wdir}')

            dwpt = _rh_to_dwpt(temp, rh)
            mixr = _dwpt_to_mixr(dwpt, pres)
            thta = _theta(temp, pres)
            thte = _theta_e(temp, dwpt, pres)
            thtv = _theta_v(temp, mixr, pres)

            rows.append({
                'icao':       icao,
                'wmo':        None,
                'stn_name':   f'EC Model {lat:+d}N {abs(lon):.0f}W',
                'lat':        float(lat),
                'lon':        float(lon),
                'valid_time': valid_time,
                'hour':       hour,
                'PRES':       float(pres),
                'HGHT':       hght,
                'TEMP':       temp,
                'DWPT':       dwpt,
                'RELH':       rh,
                'MIXR':       mixr,
                'DRCT':       wdir,
                'SPED':       wspd,
                'THTA':       thta,
                'THTE':       thte,
                'THTV':       thtv,
            })
    return rows


# ── Fetch loop with status display ────────────────────────────────────────────
ec_ua_rows   = []
ec_ua_errors = []

print(f'Fetching EC model upper-air data: {len(EC_UA_LATITUDES)} stations × '
      f'{len(EC_PRESSURE_LEVELS)} levels × {len(UA_HOURS)} synoptic hours...')
print(f'Pressure levels: {EC_PRESSURE_LEVELS} hPa\n')

for _lat in EC_UA_LATITUDES:
    _id = ec_ua_icao(_lat, EC_UA_LONGITUDE)
    print(f'  {_id}  ({_lat:+03d}°N) … ', end='', flush=True)
    try:
        _data = _fetch_ec_ua(_lat, EC_UA_LONGITUDE)
        _rows = _parse_ec_ua(_lat, EC_UA_LONGITUDE, _data)
        ec_ua_rows.extend(_rows)
        # Count per hour for status
        _h_counts = {}
        for _r in _rows:
            _h_counts[_r['hour']] = _h_counts.get(_r['hour'], 0) + 1
        _hstr = '  '.join(f"{h:02d}Z:{n}lvls" for h, n in sorted(_h_counts.items()))
        print(f'✓  {len(_rows)} rows  [{_hstr}]')
    except Exception as _e:
        print(f'✗  {_e}')
        ec_ua_errors.append(_id)

print(f'\n✓  EC upper-air rows fetched: {len(ec_ua_rows)}')
if ec_ua_errors:
    print(f'  ✗ Failed: {ec_ua_errors}')


# ── Build DataFrame matching ua_raw_df schema exactly ─────────────────────────
ec_ua_df = pd.DataFrame(ec_ua_rows, columns=[
    'icao','wmo','stn_name','lat','lon','valid_time','hour',
    'PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV'
])

print(f'\n✓ ec_ua_df shape: {ec_ua_df.shape}')
print(f'  Stations: {ec_ua_df["icao"].nunique()}')
print(f'  Valid times: {sorted(ec_ua_df["valid_time"].unique())}')
print(f'  Pressure levels: {sorted(ec_ua_df["PRES"].unique(), reverse=True)}')


# ── Merge into ua_raw_df ───────────────────────────────────────────────────────
_before = len(ua_raw_df)
ua_raw_df = pd.concat([ua_raw_df, ec_ua_df], ignore_index=True)
print(f'\n✓ ua_raw_df: {_before} → {len(ua_raw_df)} rows '
      f'(+{len(ec_ua_df)} EC model levels added)')
print(f'  Total stations: {ua_raw_df["icao"].nunique()}  '
      f'(real: {_before // max(1, len(EC_PRESSURE_LEVELS))}  '
      f'+ EC model: {ec_ua_df["icao"].nunique()})')


# ── Summary display ───────────────────────────────────────────────────────────
_summary = (
    ua_raw_df.groupby(['icao','hour','valid_time'])
    .agg(levels=('PRES', 'count'))
    .reset_index()
    .query('icao.str.startswith("ECM")', engine='python')
)


# ── Cell UA-3 . Standard-level summary table (850/700/500/250 hPa) ────────
print('--- Upper Air station - Data extract ---')
import pandas as pd
import numpy as np

STANDARD_LEVELS = [850, 700, 500, 250]
LEVEL_TOL       = 25   # hPa tolerance — use closest level within this window

FIELDS = ['PRES','HGHT','TEMP','DWPT','RELH','MIXR','DRCT','SPED','THTA','THTE','THTV']

def find_closest_level(group_df, target_p, tol=LEVEL_TOL):
    """
    From a single station/hour DataFrame (already sorted by PRES desc),
    find the row closest to target_p within ±tol hPa.
    Returns a dict of field values or all-None if nothing within tolerance.
    """
    sub = group_df.copy()
    sub['_dist'] = (sub['PRES'] - target_p).abs()
    sub = sub[sub['_dist'] <= tol]
    if sub.empty:
        return {f: None for f in FIELDS}
    best = sub.loc[sub['_dist'].idxmin()]
    return {f: best[f] if not pd.isna(best[f]) else None for f in FIELDS}

# ── Build summary ──────────────────────────────────────────────────────────
summary_rows = []

# Ensure ua_raw_df exists from Cell UA-2
if 'ua_raw_df' in globals():
    for (icao, hour), grp in ua_raw_df.groupby(['icao', 'hour'], sort=False):
        grp = grp.sort_values('PRES', ascending=False).reset_index(drop=True)

        # station meta — take from first row
        meta = grp.iloc[0]
        base = {
            'icao':       icao,
            'wmo':        meta['wmo'],
            'stn_name':   meta['stn_name'],
            'lat':        meta['lat'],
            'lon':        meta['lon'],
            'valid_time': meta['valid_time'],
            'hour':       hour,
        }

        for lvl in STANDARD_LEVELS:
            vals = find_closest_level(grp, lvl)
            for f in FIELDS:
                base[f'{f}_{lvl}'] = vals[f]

        summary_rows.append(base)

    ua_summary_df = pd.DataFrame(summary_rows).sort_values(['icao', 'hour']).reset_index(drop=True)

    # ── Only keep hours where at least one real sounding station has data ──
    # Real sounding stations are identified by source != 'ec_model' (or no source col)
    _real_src = ['bufr', 'wyoming', 'sounding', 'raob']  # adjust to match your source labels
    # Real sounding hours = hours that have at least one non-EC-model station
    # Find valid (hour, date) combos from real sounding stations only
    _real_mask = ~ua_summary_df['icao'].str.startswith('ECML')
    _real_df   = ua_summary_df[_real_mask].copy()
    _real_df['_date'] = _real_df['valid_time'].astype(str).str[:10]
    _real_hours = set(_real_df['hour'].astype(int).unique())

    # Build set of valid (hour, date) pairs from real soundings
    _valid_combos = set(
        zip(_real_df['hour'].astype(int), _real_df['_date'])
    )
    print(f'  Valid (hour, date) combos from real soundings: {sorted(_valid_combos)}')

    # Apply same filter to full dataframe
    ua_summary_df['_date'] = ua_summary_df['valid_time'].astype(str).str[:10]
    ua_summary_df['_combo'] = list(zip(ua_summary_df['hour'].astype(int), ua_summary_df['_date']))
    ua_summary_df = ua_summary_df[ua_summary_df['_combo'].isin(_valid_combos)].drop(columns=['_date','_combo']).reset_index(drop=True)
    print(f'  After date+hour filter: {len(ua_summary_df)} rows')

    if _real_hours:
        ua_summary_df = ua_summary_df[ua_summary_df['hour'].isin(_real_hours)].reset_index(drop=True)
        print(f'  Filtered to real sounding hours: {sorted(_real_hours)}')

    _ua_times = ua_summary_df.groupby('hour')['valid_time'].max().reset_index().sort_values('hour')
    _ua_date_map = {}
    for _, _row in _ua_times.iterrows():
        _ua_date_map[str(int(_row['hour']))] = str(_row['valid_time'])
        print(f'  UA available: {_row["valid_time"]}  hour={int(_row["hour"])}Z')
    print(f'✓ Summary table: {len(ua_summary_df)} rows  '\
          f'({ua_summary_df["icao"].nunique()} stations × '\
          f'{ua_summary_df["hour"].nunique()} hours)')

    # ── Coverage check ──
    missing_report = []
    for lvl in STANDARD_LEVELS:
        n_miss = ua_summary_df[f'TEMP_{lvl}'].isna().sum()
        if n_miss: missing_report.append(f'  {lvl} hPa: {n_miss} station-hours missing TEMP')
    if missing_report:
        print('⚠ Missing data:'); [print(m) for m in missing_report]
    else:
        print('✔ All standard levels populated')

# ── Display ──
    def _style_summary(df):
        s = df.style.format(na_rep='—', precision=1)

        temp_cols = [c for c in df.columns if c.startswith('TEMP_') or c.startswith('DWPT_')]
        relh_cols = [c for c in df.columns if c.startswith('RELH_')]
        sped_cols = [c for c in df.columns if c.startswith('SPED_')]
        hght_cols = [c for c in df.columns if c.startswith('HGHT_')]
        mixr_cols = [c for c in df.columns if c.startswith('MIXR_')]
        thta_cols = [c for c in df.columns if c.startswith('THTA_') or
                                               c.startswith('THTE_') or
                                               c.startswith('THTV_')]
        drct_cols = [c for c in df.columns if c.startswith('DRCT_')]
        pres_cols = [c for c in df.columns if c.startswith('PRES_')]

        for cols, cmap in [
            (temp_cols, 'RdBu_r'),
            (relh_cols, 'Blues'),
            (sped_cols, 'Purples'),
            (hght_cols, 'YlOrRd'),
            (mixr_cols, 'YlGn'),
            (thta_cols, 'RdYlBu_r'),
            (drct_cols, 'twilight'),
        ]:
            existing = [c for c in cols if c in df.columns]
            if existing:
                s = s.background_gradient(subset=existing, cmap=cmap, axis=None)

        existing_pres = [c for c in pres_cols if c in df.columns]
        if existing_pres:
            s = s.bar(subset=existing_pres, color='#9ecae1', vmin=200, vmax=900)

        return s

    _sh = _style_summary(ua_summary_df.head(10)).to_html()
    _fl = _style_summary(ua_summary_df).to_html()
else:
    print('❌ Error: ua_raw_df not found. Please run Cell UA-2 first.')

print('=' * 60)
print('  BLOCK 01 — 850/700/500 hPa Ridge & Trough Detection')
print('  Method: Row-wise find_peaks (M5) on RBF-interpolated T grids')
print('=' * 60)

# ── Cell 7.10 . 850 hPa Ridge & Trough — Method 5 (Row-wise find_peaks) ──
#
# Uses M5 (row-wise find_peaks per latitude) — selected as best method.
# Produces two clean figures per analysis hour:
#   Figure 1: WARM RIDGE  (purple lines on temperature fill)
#   Figure 2: COLD TROUGH (purple lines on temperature fill)
#
# Tune these parameters to adjust sensitivity:
M5_SIGMA      = 2.0    # Gaussian smooth before peak detection
M5_PROMINENCE = 0.5    # minimum peak prominence (°C) — raise to reduce noise
M5_WIDTH      = 2      # minimum peak width (grid cells)
M5_MAX_PEAKS  = 4      # max peaks kept per latitude row (strongest only)
M5_MAX_DIST   = 4.0    # max gap (deg) between points to stay in same segment
M5_MIN_PTS    = 4      # minimum points to form a valid segment
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════
DARK         = '#0d1117'
PANEL        = '#161b22'
WHITE        = '#e6edf3'
RIDGE_COLOR  = '#CC44FF'
TROUGH_COLOR = '#CC44FF'
RIDGE_LW     = 3.2
TROUGH_LW    = 3.2

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _rbf_grid(la, lo, va, N=180, pad=1.5, sigma=1.5):
    seen = {}
    for a, o, v in zip(la, lo, va):
        seen.setdefault((round(a,2), round(o,2)), []).append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in seen.items()]
    if len(pts) < 8:
        return None, None, None
    la2 = np.array([p[0] for p in pts])
    lo2 = np.array([p[1] for p in pts])
    va2 = np.array([p[2] for p in pts])
    lv  = np.linspace(lo2.min()-pad, lo2.max()+pad, N)
    ltv = np.linspace(la2.min()-pad, la2.max()+pad, N)
    GL, GLA = np.meshgrid(lv, ltv)
    try:
        rbf  = RBFInterpolator(np.column_stack([lo2, la2]), va2,
                               kernel='thin_plate_spline',
                               smoothing=max(0.3*len(pts), 1e-6))
        grid = rbf(np.column_stack([GL.ravel(), GLA.ravel()])).reshape(N, N)
    except Exception:
        return None, None, None
    return gaussian_filter(grid, sigma=sigma), lv, ltv


def _build_t850(df_hr):
    rows = df_hr.dropna(subset=['TEMP_850','lat','lon'])
    if len(rows) < 8:
        return None, None, None
    return _rbf_grid(rows['lat'].values, rows['lon'].values,
                     rows['TEMP_850'].values, sigma=1.5)

def _build_tlvl(df_hr, lvl):
    col = f'TEMP_{lvl}'
    rows = df_hr.dropna(subset=[col,'lat','lon'])
    if len(rows) < 8:
        return None, None, None
    return _rbf_grid(rows['lat'].values, rows['lon'].values,
                     rows[col].values, sigma=1.5)


def _build_slp(recs):
    pts = [(d['lat'], d['lon'], d['slp'])
           for d in recs if d.get('slp') is not None]
    if len(pts) < 8:
        return None, None, None
    la = np.array([p[0] for p in pts])
    lo = np.array([p[1] for p in pts])
    va = np.array([p[2] for p in pts])
    return _rbf_grid(la, lo, va, sigma=3.0)


def _pts_to_segs(pts, max_dist=M5_MAX_DIST, min_pts=M5_MIN_PTS):
    """Union-find segment builder from (lat, lon, strength) list."""
    if len(pts) < min_pts:
        return []
    arr = np.array(pts)

    dedup = {}
    for r in arr:
        k = (round(r[0]*2)/2, round(r[1]*2)/2)
        if k not in dedup or r[2] > dedup[k][2]:
            dedup[k] = r
    arr = np.array(list(dedup.values()))
    if len(arr) < min_pts:
        return []

    tree   = cKDTree(arr[:,:2])
    pairs  = tree.query_pairs(max_dist)
    parent = list(range(len(arr)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = _find(a), _find(b)
        if ra != rb: parent[ra] = rb

    clusters = {}
    for i in range(len(arr)):
        clusters.setdefault(_find(i), []).append(i)

    segs = []
    for idxs in clusters.values():
        if len(idxs) < min_pts: continue
        sp = sorted([(arr[i][0], arr[i][1]) for i in idxs], key=lambda p: p[0])
        sub, cur = [], [sp[0]]
        for k in range(1, len(sp)):
            if (abs(sp[k][0]-sp[k-1][0]) > max_dist*1.5 or
                abs(sp[k][1]-sp[k-1][1]) > max_dist*1.5):
                if len(cur) >= min_pts: sub.append(cur)
                cur = []
            cur.append(sp[k])
        if len(cur) >= min_pts: sub.append(cur)
        segs.extend(sub)
    return segs


# ══════════════════════════════════════════════════════════════════════════
# M5 DETECTOR
# ══════════════════════════════════════════════════════════════════════════
def detect_m5(T, lv, ltv):
    """
    Row-wise find_peaks per latitude band.
    Ridge  → local T maxima  (peaks of  T)
    Trough → local T minima  (peaks of -T)
    Returns (ridge_segs, trough_segs)
    """
    Ts = gaussian_filter(T, sigma=M5_SIGMA)
    ridge_pts, trough_pts = [], []

    for j, lat in enumerate(ltv):
        row = Ts[j, :]

        pk, pr = find_peaks(row, prominence=M5_PROMINENCE, width=M5_WIDTH)
        if len(pk):
            top = np.argsort(pr['prominences'])[-M5_MAX_PEAKS:]
            for idx in pk[top]:
                prom = pr['prominences'][np.where(pk == idx)[0][0]]
                ridge_pts.append((lat, lv[idx], float(prom)))

        tr, tpr = find_peaks(-row, prominence=M5_PROMINENCE, width=M5_WIDTH)
        if len(tr):
            top = np.argsort(tpr['prominences'])[-M5_MAX_PEAKS:]
            for idx in tr[top]:
                prom = tpr['prominences'][np.where(tr == idx)[0][0]]
                trough_pts.append((lat, lv[idx], float(prom)))

    return _pts_to_segs(ridge_pts), _pts_to_segs(trough_pts)


# ══════════════════════════════════════════════════════════════════════════
# PLOT ONE FIGURE  (mode = 'ridge' | 'trough')
# ══════════════════════════════════════════════════════════════════════════
def _make_figure(mode, segs, T850, lv, ltv,
                 slp_grid, slp_lv, slp_ltv,
                 df_hr, valid_str):

    is_ridge   = (mode == 'ridge')
    line_color = RIDGE_COLOR if is_ridge else TROUGH_COLOR
    line_lw    = RIDGE_LW    if is_ridge else TROUGH_LW
    line_ls    = 'solid'     if is_ridge else (0, (6, 3))
    mode_title = '🟣 WARM RIDGE' if is_ridge else '🟣 COLD TROUGH'
    hdr_color  = '#DD88FF'

    fig = plt.figure(figsize=(16, 10), facecolor=DARK)
    fig.suptitle(
        f'850 hPa  {mode_title}  ·  M5 Row-wise find_peaks  ·  {valid_str}',
        fontsize=16, fontweight='bold', color=hdr_color, y=0.982)

    ax = fig.add_axes([0.06, 0.07, 0.87, 0.87], facecolor=PANEL)
    GL, GLA = np.meshgrid(lv, ltv)

    cf = ax.contourf(GL, GLA, T850, levels=30, cmap='RdYlBu_r', alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.022, pad=0.01)
    cbar.set_label('850 hPa T (°C)', color=WHITE, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=WHITE, labelsize=8)
    plt.setp(cbar.ax.get_yticklabels(), color=WHITE)
    cbar.outline.set_edgecolor('#555')

    t_lvls = np.arange(np.floor(T850.min()/2)*2,
                       np.ceil( T850.max()/2)*2 + 2, 2)
    ax.contour(GL, GLA, T850, levels=t_lvls,
               colors='white', linewidths=0.35, alpha=0.22)

    if slp_grid is not None:
        SL, SLA = np.meshgrid(slp_lv, slp_ltv)
        s_min = np.floor(slp_grid.min()/SLP_INTERVAL)*SLP_INTERVAL
        s_max = np.ceil( slp_grid.max()/SLP_INTERVAL)*SLP_INTERVAL
        slvls = np.arange(s_min, s_max + SLP_INTERVAL, SLP_INTERVAL)
        ax.contour(SL, SLA, slp_grid, levels=slvls,
                   colors='white', alpha=0.50,
                   linewidths=[1.6 if int(l)%20==0 else 0.6 for l in slvls])

    for _, row in df_hr.iterrows():
        ax.scatter(row['lon'], row['lat'],
                   s=16, color=WHITE, alpha=0.6, zorder=6, linewidths=0)
        ax.text(row['lon']+0.25, row['lat']+0.25, row['icao'],
                fontsize=5.5, color=WHITE, alpha=0.55,
                path_effects=[pe.withStroke(linewidth=1, foreground=DARK)])

    total_pts = 0
    for seg in segs:
        xs = [p[1] for p in seg]
        ys = [p[0] for p in seg]
        total_pts += len(seg)
        ax.plot(xs, ys,
                color=line_color, linewidth=line_lw, linestyle=line_ls,
                solid_capstyle='round', solid_joinstyle='round',
                zorder=10,
                path_effects=[
                    pe.Stroke(linewidth=line_lw+3.0, foreground=DARK, alpha=0.8),
                    pe.Normal()])

        mi = len(seg) // 2
        lbl = 'RIDGE' if is_ridge else 'TRGH'
        ax.text(seg[mi][1], seg[mi][0], lbl,
                fontsize=6.5, color=line_color, fontweight='bold',
                ha='center', va='bottom' if is_ridge else 'top',
                zorder=11,
                path_effects=[pe.withStroke(linewidth=2, foreground=DARK)])

    seg_lbl = (f"{len(segs)} segment{'s' if len(segs)!=1 else ''}  "
               f"·  {total_pts} points")
    handles = [
        Line2D([0],[0], color='white',      lw=0.8,          label='850 hPa T isotherms'),
        Line2D([0],[0], color='white',      lw=1.5, alpha=0.5, label='SLP isobars'),
        Line2D([0],[0], color=line_color,   lw=2.8, ls=line_ls,
               label=f"{'Ridge' if is_ridge else 'Trough'} — {seg_lbl}"),
    ]
    ax.legend(handles=handles, loc='lower left',
              fontsize=10, facecolor='#1c2333', edgecolor='#555',
              labelcolor=WHITE, framealpha=0.94, handlelength=3.2)

    all_lons = df_hr['lon'].dropna().values
    all_lats = df_hr['lat'].dropna().values
    if len(all_lons):
        ax.set_xlim(all_lons.min()-2, all_lons.max()+2)
        ax.set_ylim(all_lats.min()-1, all_lats.max()+3)

    ax.set_xlabel('Longitude (°E)', color=WHITE, fontsize=10)
    ax.set_ylabel('Latitude (°N)',  color=WHITE, fontsize=10)
    ax.tick_params(colors=WHITE, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    fig.text(0.5, 0.005,
             f'M5 params — σ={M5_SIGMA}  prominence={M5_PROMINENCE}  '
             f'width={M5_WIDTH}  max_peaks/row={M5_MAX_PEAKS}  '
             f'max_link_dist={M5_MAX_DIST}°  min_seg_pts={M5_MIN_PTS}',
             ha='center', fontsize=7.5, color='#666', style='italic')

    return fig


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
_hours = sorted(ua_summary_df['hour'].unique())

for _hr in _hours:
    _df_hr = ua_summary_df[ua_summary_df['hour'] == _hr].copy()

    # ── Fix: isolate the dominant valid_time for this hour ─────────────
    _dominant_vt = _df_hr['valid_time'].value_counts().idxmax()
    _stale = _df_hr[_df_hr['valid_time'] != _dominant_vt]
    if len(_stale):
        print(f'  ⚠ Dropping {len(_stale)} stale-date rows '
              f'({_stale["valid_time"].iloc[0]}) — '
              f'kept dominant: {_dominant_vt}')
        print(f'    Stale stations: {sorted(_stale["icao"].unique().tolist())}')
    _df_hr = _df_hr[_df_hr['valid_time'] == _dominant_vt].copy()
    _valid = _dominant_vt
    # ───────────────────────────────────────────────────────────────────

    print(f'\n── {_valid} ──')

    T850, lv, ltv = _build_t850(_df_hr)
    if T850 is None:
        print('  ⚠ insufficient 850 hPa T — skipping'); continue

    _ts_keys = sorted(set(d.get('timestamp','') for d in metar_records))
    _best_ts = next(
        (t for t in _ts_keys
         if f'{int(_hr):02d}' in str(t) or str(int(_hr)) in str(t)),
        _ts_keys[0] if _ts_keys else None)
    _slp_recs = ([d for d in metar_records if d.get('timestamp') == _best_ts]
                 if _best_ts else metar_records)
    slp_grid, slp_lv, slp_ltv = _build_slp(_slp_recs)

    print('  Detecting ridges and troughs (M5)…', end=' ', flush=True)
    ridge_segs, trough_segs = detect_m5(T850, lv, ltv)
    print(f'ridge: {len(ridge_segs)} seg  |  trough: {len(trough_segs)} seg')

    fig1 = _make_figure('ridge', ridge_segs, T850, lv, ltv,
                        slp_grid, slp_lv, slp_ltv, _df_hr, _valid)
    f1 = f'ridge_{int(_hr):02d}Z.png'
    fig1.savefig(f1, dpi=160, bbox_inches='tight', facecolor=DARK)
    plt.show()
    print(f'  ✅ {f1}')

    fig2 = _make_figure('trough', trough_segs, T850, lv, ltv,
                        slp_grid, slp_lv, slp_ltv, _df_hr, _valid)
    f2 = f'trough_{int(_hr):02d}Z.png'
    fig2.savefig(f2, dpi=160, bbox_inches='tight', facecolor=DARK)
    plt.show()
    print(f'  ✅ {f2}')

    globals()[f'ridge_segs_{int(_hr):02d}']  = ridge_segs
    globals()[f'trough_segs_{int(_hr):02d}'] = trough_segs

    T700, lv700, ltv700 = _build_tlvl(_df_hr, 700)
    if T700 is not None:
        ridge_segs_700, trough_segs_700 = detect_m5(T700, lv700, ltv700)
        print(f'  700 hPa — ridge: {len(ridge_segs_700)} seg  |  trough: {len(trough_segs_700)} seg')
    else:
        ridge_segs_700, trough_segs_700 = [], []
        print('  700 hPa — insufficient data')
    globals()[f'ridge_segs_700_{int(_hr):02d}']  = ridge_segs_700
    globals()[f'trough_segs_700_{int(_hr):02d}'] = trough_segs_700

    T500, lv500, ltv500 = _build_tlvl(_df_hr, 500)
    if T500 is not None:
        ridge_segs_500, trough_segs_500 = detect_m5(T500, lv500, ltv500)
        print(f'  500 hPa — ridge: {len(ridge_segs_500)} seg  |  trough: {len(trough_segs_500)} seg')
    else:
        ridge_segs_500, trough_segs_500 = [], []
        print('  500 hPa — insufficient data')
    globals()[f'ridge_segs_500_{int(_hr):02d}']  = ridge_segs_500
    globals()[f'trough_segs_500_{int(_hr):02d}'] = trough_segs_500

print('\n✅ Block 01 complete — ridge/trough segments stored in globals()')

# ──SFC Smoothing controls ──────────────────────────────────
TMP_RBF_SMOOTHING = 0.05   # was 0.20
TMP_SIGMA         = 1.0    # was 2.5
TTD_RBF_SMOOTHING = 0.100   # was 0.20
TTD_SIGMA         = 1.0    # was 2.5

print('=' * 60)
print('  BLOCK 02 — RBF Grid Interpolation')
print('  Builds SLP, Temperature, and T-Td grids from METAR obs')
print('=' * 60)

# ── Cell 6 . Kriging / RBF interpolation ──────────────────────
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
import numpy as np






def build_grid(records, field, method='rbf', N=220, pad=1.5,
               rbf_smoothing=0.3, sigma=3.0):
    '''
    Interpolate scattered obs onto a regular grid.
    method: 'rbf' uses thin-plate spline (smooth, fast)
            'kriging' uses Ordinary Kriging (best for SLP)
    Returns (grid_2d, lon_vec, lat_vec, lons_flat, lats_flat)
    '''
    pts = [(d['lat'], d['lon'], d[field])
           for d in records if d.get(field) is not None]
    if len(pts) < 8:
        return None, None, None, None, None

    _seen = {}
    for la, lo, v in pts:
        key = (round(la, 2), round(lo, 2))
        if key not in _seen:
            _seen[key] = []
        _seen[key].append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in _seen.items()]
    if len(pts) < 8:
        return None, None, None, None, None

    lats = np.array([p[0] for p in pts])
    lons = np.array([p[1] for p in pts])
    vals = np.array([p[2] for p in pts], dtype=float)

    lat_min, lat_max = lats.min()-pad, lats.max()+pad
    lon_min, lon_max = lons.min()-pad, lons.max()+pad

    lon_vec = np.linspace(lon_min, lon_max, N)
    lat_vec = np.linspace(lat_min, lat_max, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)

    if method == 'kriging':
        ok = OrdinaryKriging(
            lons, lats, vals,
            variogram_model='linear',
            verbose=False, enable_plotting=False
        )
        z, _ = ok.execute('grid', lon_vec, lat_vec)
        grid = np.array(z)
    else:
        obs_xy = np.column_stack([lons, lats])
        try:
            rbf = RBFInterpolator(
                obs_xy, vals,
                kernel='thin_plate_spline',
                smoothing=max(rbf_smoothing * len(pts), 1e-6)
            )
        except np.linalg.LinAlgError:
            rbf = RBFInterpolator(
                obs_xy, vals,
                kernel='linear',
                smoothing=max(rbf_smoothing * len(pts), 1.0)
            )
        qi = np.column_stack([glon.ravel(), glat.ravel()])
        grid = rbf(qi).reshape(N, N)

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    return grid, lon_vec, lat_vec, lons, lats


print(f'Building SLP grid ({INTERP_METHOD})...')
slp_grid, lon_vec, lat_vec, obs_lons, obs_lats = build_grid(
    metar_records, 'slp',
    method=INTERP_METHOD, N=GRID_N,
    rbf_smoothing=RBF_SMOOTHING, sigma=SIGMA_SMOOTH
)
if slp_grid is not None:
    print(f'  ✓ SLP grid: {slp_grid.shape}  '
          f'range {slp_grid.min():.1f}–{slp_grid.max():.1f} hPa')
else:
    print('  ⚠ Not enough SLP data')

print('Building temperature grid...')
tmp_grid, tlon_vec, tlat_vec, _, _ = build_grid(
    metar_records, 'temp',
    method='rbf', N=GRID_N,
    rbf_smoothing=TMP_RBF_SMOOTHING, sigma=TMP_SIGMA
)
if tmp_grid is not None:
    print(f'  ✓ Temp grid: range {tmp_grid.min():.1f}–{tmp_grid.max():.1f} °C')

print('Building T-Td spread grid...')
ttd_grid, ttd_lon_vec, ttd_lat_vec, _, _ = build_grid(
    metar_records, 't_td',
    method='rbf', N=GRID_N,
    rbf_smoothing=TTD_RBF_SMOOTHING, sigma=TTD_SIGMA
)
if ttd_grid is not None:
    print(f'  ✓ T-Td grid: range {ttd_grid.min():.1f}–{ttd_grid.max():.1f} °C')

print('Building temperature contour grid...')
temp_grid, temp_lon_vec, temp_lat_vec, _, _ = build_grid(
    metar_records, 'temp',
    method='rbf', N=GRID_N,
    rbf_smoothing=TMP_RBF_SMOOTHING, sigma=TMP_SIGMA
)
if temp_grid is not None:
    print(f'  ✓ Temp contour grid: range {temp_grid.min():.1f}–{temp_grid.max():.1f} °C')

print('\n✅ Block 02 complete — grids: slp_grid, tmp_grid, ttd_grid, temp_grid')

print('=' * 60)
print('  BLOCK 03 — H/L Pressure Centre Detection')
print('  Locates surface High and Low centres from SLP grid')
print('=' * 60)

import math
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, label, gaussian_filter


def find_hl_centers(grid, lon_vec, lat_vec,
                    neighborhood=20, min_delta=2.0):
    '''
    Find local maxima (H) and minima (L) in grid.
    neighborhood: search radius in grid cells
    min_delta:    minimum difference from background to qualify
    Returns list of dicts: {type, lat, lon, val}
    '''
    sg    = gaussian_filter(grid, sigma=HL_SIGMA)
    max_f = maximum_filter(sg, size=neighborhood)
    min_f = minimum_filter(sg, size=neighborhood)

    is_max = (sg == max_f) & (sg - min_f > min_delta)
    is_min = (sg == min_f) & (max_f - sg > min_delta)

    centers = []
    for typ, mask in [('H', is_max), ('L', is_min)]:
        lbl, n = label(mask)
        for i in range(1, n+1):
            rows, cols = np.where(lbl == i)
            best = np.argmax(sg[rows, cols]) if typ == 'H' else np.argmin(sg[rows, cols])
            r, c = rows[best], cols[best]

            if r < neighborhood or r > len(lat_vec)-neighborhood: continue
            if c < neighborhood or c > len(lon_vec)-neighborhood: continue

            _grid_val = float(grid[r, c])

            def _grid_at(sta_lat, sta_lon):
                _ri = int(round((sta_lat - lat_vec[0]) /
                                (lat_vec[-1] - lat_vec[0]) * (len(lat_vec) - 1)))
                _ci = int(round((sta_lon - lon_vec[0]) /
                                (lon_vec[-1] - lon_vec[0]) * (len(lon_vec) - 1)))
                _ri = max(0, min(len(lat_vec) - 1, _ri))
                _ci = max(0, min(len(lon_vec) - 1, _ci))
                return float(grid[_ri, _ci])

            _thresh = SLP_INTERVAL

            if typ == 'H':
                _inside = [
                    d['slp'] for d in metar_records
                    if d['slp'] is not None
                    and _grid_at(d['lat'], d['lon']) >= _grid_val - _thresh
                ]
                _val = (math.floor(max(_inside)) + 1) if _inside else (math.floor(_grid_val) + 1)
            else:
                _inside = [
                    d['slp'] for d in metar_records
                    if d['slp'] is not None
                    and _grid_at(d['lat'], d['lon']) <= _grid_val + _thresh
                ]
                _val = (math.ceil(min(_inside)) - 1) if _inside else (math.ceil(_grid_val) - 1)

            centers.append(dict(
                type=typ,
                lat=lat_vec[r], lon=lon_vec[c],
                val=float(_val)
            ))
    return centers


if slp_grid is not None:
    hl_centers = find_hl_centers(slp_grid, lon_vec, lat_vec,
                                 neighborhood=HL_NEIGHBORHOOD,
                                 min_delta=HL_MIN_DELTA)
    highs = [c for c in hl_centers if c['type'] == 'H']
    lows  = [c for c in hl_centers if c['type'] == 'L']
    print(f'  ✓ Found {len(highs)} High(s) and {len(lows)} Low(s)')
    for c in hl_centers:
        print(f"    {c['type']}  {c['val']:.1f} hPa  "
              f"@ {c['lat']:.2f}°N  {c['lon']:.2f}°E")
else:
    hl_centers = []
    print('  ⚠ No SLP grid — H/L detection skipped')

print('\n✅ Block 03 complete — hl_centers list ready')

print('=' * 60)
print('  BLOCK 04 — Per-Timestamp SLP Contours')
print('  Builds SLP, T-Td, and Temperature contours for each METAR time')
print('=' * 60)
print(f"TMP_RBF_SMOOTHING: {TMP_RBF_SMOOTHING}")
print(f"TMP_SIGMA: {TMP_SIGMA}")
print(f"TTD_RBF_SMOOTHING: {TTD_RBF_SMOOTHING}")
print(f"TTD_SIGMA: {TTD_SIGMA}")


import json as _json_ts
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

_ts_all_pre = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
_ts_slp = {}

for _ts in _ts_all_pre:
    _recs = [d for d in metar_records if d['timestamp'] == _ts]

    # ── SLP grid ──
    _grid, _lv, _ltv, _, _ = build_grid(
        _recs, 'slp',
        method=INTERP_METHOD, N=GRID_N,
        rbf_smoothing=RBF_SMOOTHING, sigma=SIGMA_SMOOTH
    )

    if _grid is None:
        _ts_slp[_ts] = {'contours': [], 'hl': []}
        print(f'  {_ts}: insufficient SLP data, skipped')
        continue

    # ── SLP contours ──
    _glon, _glat = np.meshgrid(_lv, _ltv)
    _slp_min = np.floor(_grid.min() / SLP_INTERVAL) * SLP_INTERVAL
    _slp_max = np.ceil(_grid.max()  / SLP_INTERVAL) * SLP_INTERVAL
    _levels  = np.arange(_slp_min, _slp_max + SLP_INTERVAL, SLP_INTERVAL)
    _fig, _ax = plt.subplots(figsize=(1, 1))
    _cs = _ax.contour(_glon, _glat, _grid, levels=_levels)
    plt.close(_fig)

    _contours = []
    for _li, _lvl in enumerate(_cs.levels):
        _is_major = (int(_lvl) % 20 == 0)
        _weight   = 2.5 if _is_major else (1.4 if int(_lvl) % 8 == 0 else 0.7)
        _opacity  = 0.95 if _is_major else (0.65 if int(_lvl) % 8 == 0 else 0.40)
        for _coords in _cs.allsegs[_li]:
            if len(_coords) < 2: continue
            _mid = _coords[len(_coords) // 2]
            _contours.append({
                'level':     float(_lvl),
                'weight':    _weight,
                'opacity':   _opacity,
                'coords':    [[float(c[0]), float(c[1])] for c in _coords],
                'label_lon': float(_mid[0]),
                'label_lat': float(_mid[1]),
            })

    # ── H/L centres ──
    _hl = find_hl_centers(_grid, _lv, _ltv,
                          neighborhood=HL_NEIGHBORHOOD,
                          min_delta=HL_MIN_DELTA)

    # ── T-Td contours ──
    _ttd_check = [d.get('temp') - d.get('dew') for d in _recs if d.get('temp') is not None and d.get('dew') is not None]
    print(f'  {_ts}: T-Td samples: {len(_ttd_check)}, e.g. {_ttd_check[:3]}')
    for _r in _recs:
        if _r.get('temp') is not None and _r.get('dew') is not None:
            _r['_ttd'] = _r['temp'] - _r['dew']
        else:
            _r['_ttd'] = None
    _ttd_grid, _ttd_lv, _ttd_ltv, _, _ = build_grid(
        _recs, '_ttd',
        method='rbf', N=GRID_N,
        rbf_smoothing=TTD_RBF_SMOOTHING, sigma=TTD_SIGMA
    )
    _ttd_contours = []
    if _ttd_grid is not None:
        _TTD_INTERVAL = 2.0
        _ttd_min = np.floor(_ttd_grid.min() / _TTD_INTERVAL) * _TTD_INTERVAL
        _ttd_max = np.ceil(_ttd_grid.max()  / _TTD_INTERVAL) * _TTD_INTERVAL
        _ttd_levels = np.arange(_ttd_min, _ttd_max + _TTD_INTERVAL, _TTD_INTERVAL)
        _glon_t, _glat_t = np.meshgrid(_ttd_lv, _ttd_ltv)
        _fig_t, _ax_t = plt.subplots(figsize=(1, 1))
        _cs_t = _ax_t.contour(_glon_t, _glat_t, _ttd_grid, levels=_ttd_levels)
        plt.close(_fig_t)
        for _li_t, _lvl_t in enumerate(_cs_t.levels):
            for _coords_t in _cs_t.allsegs[_li_t]:
                if len(_coords_t) < 2: continue
                _mid_t = _coords_t[len(_coords_t) // 2]
                _ttd_contours.append({
                    'level':     float(_lvl_t),
                    'coords':    [[float(c[0]), float(c[1])] for c in _coords_t],
                    'label_lon': float(_mid_t[0]),
                    'label_lat': float(_mid_t[1]),
                })

    # ── Temperature contours ──
    _tmp_grid2, _tmp_lv2, _tmp_ltv2, _, _ = build_grid(
        _recs, 'temp',
        method='rbf', N=GRID_N,
        rbf_smoothing=TMP_RBF_SMOOTHING, sigma=TMP_SIGMA
    )
    _tmp_contours = []
    if _tmp_grid2 is not None:
        _TMP_INTERVAL = 2.0
        _tmp_min = np.floor(_tmp_grid2.min() / _TMP_INTERVAL) * _TMP_INTERVAL
        _tmp_max = np.ceil(_tmp_grid2.max()  / _TMP_INTERVAL) * _TMP_INTERVAL
        _tmp_levels = np.arange(_tmp_min, _tmp_max + _TMP_INTERVAL, _TMP_INTERVAL)
        _glon_m, _glat_m = np.meshgrid(_tmp_lv2, _tmp_ltv2)
        _fig_m, _ax_m = plt.subplots(figsize=(1, 1))
        _cs_m = _ax_m.contour(_glon_m, _glat_m, _tmp_grid2, levels=_tmp_levels)
        plt.close(_fig_m)
        for _li_m, _lvl_m in enumerate(_cs_m.levels):
            for _coords_m in _cs_m.allsegs[_li_m]:
                if len(_coords_m) < 2: continue
                _mid_m = _coords_m[len(_coords_m) // 2]
                _tmp_contours.append({
                    'level':     float(_lvl_m),
                    'coords':    [[float(c[0]), float(c[1])] for c in _coords_m],
                    'label_lon': float(_mid_m[0]),
                    'label_lat': float(_mid_m[1]),
                })

    _ts_slp[_ts] = {
        'contours':     _contours,
        'hl':           _hl,
        'ttd_contours': _ttd_contours,
        'tmp_contours': _tmp_contours,
    }
    print(f'  {_ts}: {len(_contours)} SLP segs, '
          f'{len(_ttd_contours)} T-Td segs, '
          f'{len(_tmp_contours)} Temp segs, '
          f'{sum(1 for x in _hl if x["type"]=="H")} H, '
          f'{sum(1 for x in _hl if x["type"]=="L")} L')

_ts_slp_json_str = _json_ts.dumps(_ts_slp)
print(f'\n✅ Block 04 complete — per-timestamp SLP/HL ready for {len(_ts_slp)} timestamps')

# ── Install MetPy if not present ──────────────────────────────────────────
try:
    import metpy
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'metpy', '-q'])



print('=' * 60)
print('  BLOCK 05 — Upper-Air Contours (850/700/500/250 hPa)')
print('  Height, Temp, T-Td, Wind Speed + Temperature Band Fills')
print('  H/L detection via MetPy peak_persistence')
print('  Normal temperature band highlighted green by date')
print('=' * 60)

import io as _io
import json as _json_ua
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as _mpatches
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from datetime import date as _date

from metpy.calc import peak_persistence, smooth_gaussian
from metpy.units import units


# ══════════════════════════════════════════════════════════════════════════
#  ██████  TUNING PARAMETERS — edit freely  ██████
# ══════════════════════════════════════════════════════════════════════════

# ── Interpolation / smoothing ─────────────────────────────────────────────
# RBF smoothing factor per field (larger = smoother, less faithful to obs)
_RBF_SMOOTH = {
    'HGHT': 0.15,
    'TEMP': 0.20,
    'TTDP': 0.25,
    'SPED': 0.30,
}

# Gaussian sigma for post-RBF smoothing (grid cells; ~180-cell grid over domain)
_SIGMA = {
    'HGHT': 1.5,
    'TEMP': 2.0,
    'TTDP': 2.5,
    'SPED': 2.0,
}

# Instability (T700-T500) smoothing sigma
sigmaT700500 = 2.0

# ── H/L detection — MetPy peak_persistence ───────────────────────────────
# Levels to detect H/L on
HL_LEVELS = [850, 700, 500]

# MetPy smooth_gaussian passes (n): higher = more smoothing before detection
# Synoptic-scale typical: 4–8. Increase if too many spurious centres.
HL_SMOOTH_N = 5

# Minimum persistence (metres for height field).
# inf = only the global extremum survives (too strict).
# 20–60 m is a good synoptic range. Lower = more Ls found.
HL_MIN_PERSISTENCE = {
    850: 5.0,
    700: 5.0,
    500: 10.0,
}

# Minimum separation between two H or two L centres (km).
# Reduce to catch closely-spaced systems.
HL_MIN_DISTANCE_KM = {
    850: 250.0,
    700: 280.0,
    500: 300.0,
}

# Skip centres this many degrees inside the domain edge (avoids boundary artefacts)
HL_EDGE_SKIP_DEG = 2.5



# ── W/C detection — warm/cold thermal centres ────────────────────────────
# Pressure levels to detect W/C on
WC_LEVELS = [850, 700, 500]

# Same gaussian smoothing pass count as H/L
WC_SMOOTH_N = 5

# Minimum persistence in °C — lower = more centres found
# Typical synoptic range: 1–3 °C
WC_MIN_PERSISTENCE = {
    850: 1.0,
    700: 1.0,
    500: 1.0,
}

# Minimum separation between two W or two C centres (km)
WC_MIN_DISTANCE_KM = {
    850: 400.0,
    700: 400.0,
    500: 400.0,
}

# Skip centres this many degrees inside the domain edge
WC_EDGE_SKIP_DEG = 2.5


# ── Contour intervals ─────────────────────────────────────────────────────
_INTERVALS = {
    'HGHT': 6.0,   # dam — fixed levels used instead for most levels
    'TEMP': 2.0,   # °C
    'TTDP': 2.0,   # °C
    'SPED': 5.0,   # m/s
}

# Fixed geopotential height contour levels per pressure level
UA_HGHT_LEVELS = {
    850: np.arange(1140, 1650, 30),
    700: np.arange(2520, 3180, 60),
    500: np.arange(4800, 6000, 60),
    250: None,   # auto-spaced
}

# ── Temperature band fills ────────────────────────────────────────────────
UA_TEMP_BAND_OPACITY  = 0.25
UA_TEMP_BAND_SHOW     = True


# ══════════════════════════════════════════════════════════════════════════
#  DATE-AWARE NORMAL BAND TABLES  (no user edits needed below)
# ══════════════════════════════════════════════════════════════════════════

_NORMALS_850 = [
    ( 1,  1,  -6,  -8), ( 1,  4,  -8, -10), ( 1, 15,  -6,  -8),
    ( 1, 18,  -4,  -6), ( 1, 24,  -6,  -8), ( 1, 31,  -8, -10),
    ( 2,  4,  -6,  -8), ( 3,  9,  -4,  -6), ( 3, 12,  -2,  -4),
    ( 4,  3,   0,  -2), ( 4,  5,   2,   0), ( 4,  8,   4,   2),
    ( 5,  2,   6,   4), ( 5, 11,   8,   6), ( 5, 23,  10,   8),
    ( 6,  1,  12,  10), ( 6, 27,  14,  12), ( 8, 24,  12,  10),
    ( 9,  5,  10,   8), ( 9, 17,   8,   6), (10,  1,   6,   4),
    (10, 11,   4,   2), (10, 25,   2,   0), (10, 29,   0,  -2),
    (11,  8,  -2,  -4), (11, 12,   0,  -2), (11, 17,  -2,  -4),
    (11, 26,  -4,  -6), (12,  2,  -6,  -8),
]

_NORMALS_500 = [
    ( 1,  1, -28, -30), ( 1, 15, -26, -28), ( 1, 24, -28, -30),
    ( 2, 23, -30, -32), ( 2, 27, -28, -30), ( 3,  9, -26, -28),
    ( 3, 12, -28, -30), ( 4,  5, -26, -28), ( 4, 19, -24, -26),
    ( 4, 27, -22, -24), ( 5, 11, -20, -22), ( 5, 23, -18, -20),
    ( 6, 13, -16, -18), ( 6, 27, -14, -16), ( 8,  4, -12, -14),
    ( 8, 11, -14, -16), ( 8, 31, -16, -18), ( 9, 17, -18, -20),
    (10,  3, -20, -22), (10, 21, -22, -24), (10, 29, -24, -26),
    (11, 17, -26, -28), (12,  2, -28, -30),
]

_HEIGHT_CONTROL = {
    "Jan 1":5400, "Apr 3":5460, "Apr 19":5520, "May 11":5580, "May 30":5640,
    "Jun 27":5700, "Jul 26":5760, "Aug 7":5700, "Aug 31":5640,
    "Oct 1":5580, "Oct 17":5520, "Oct 29":5460, "Nov 17":5400,
}

_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def _doy(month, day):
    try:
        return _date(2001, month, day).timetuple().tm_yday
    except ValueError:
        return 999


def _get_normal_band(pressure_level, today=None):
    if today is None:
        today = _date.today()
    table = _NORMALS_850 if pressure_level == 850 else _NORMALS_500
    today_doy = today.timetuple().tm_yday
    best_hi, best_lo, best_doy = None, None, -1
    for (m, d, hi, lo) in table:
        entry_doy = _doy(m, d)
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy
            best_hi, best_lo = hi, lo
    if best_hi is None:
        best_hi, best_lo = table[-1][2], table[-1][3]
    return best_hi, best_lo


_UA_TEMP_BANDS_BASE = [

    ( 24,  22, '#8800cc'), ( 22,  20, '#ffffff'), ( 20,  18, '#aaaaaa'),
    ( 18,  16, '#ffffff'), ( 16,  14, '#8b4513'),
    ( 14,  12, '#ffffff'), ( 12,  10, '#ffc0cb'), ( 10,   8, '#ffffff'),
    (  8,   6, '#e35335'), (  6,   4, '#ffffff'), (  4,   2, '#ff8c00'),
    (  2,   0, '#ffffff'), (  0,  -2, '#ffff00'), ( -2,  -4, '#ffffff'),
    ( -4,  -6, '#00cc00'), ( -6,  -8, '#ffffff'), ( -8, -10, '#add8e6'),
    (-10, -12, '#ffffff'), (-12, -14, '#0066ff'), (-14, -16, '#ffffff'),
    (-16, -18, '#8800cc'), (-18, -20, '#ffffff'), (-20, -22, '#aaaaaa'),
    (-22, -24, '#ffffff'), (-24, -26, '#8b4513'), (-26, -28, '#ffffff'),
]
_STATIC_GREEN_LO = -6


def _make_ua_temp_bands(pressure_level, today=None):
    if today is None:
        today = _date.today()
    normal_hi, normal_lo = _get_normal_band(pressure_level, today)
    shift = normal_lo - _STATIC_GREEN_LO
    return [(bhi + shift, blo + shift, col) for (bhi, blo, col) in _UA_TEMP_BANDS_BASE]


_TODAY = _date.today()
_normal_850_hi, _normal_850_lo = _get_normal_band(850, _TODAY)
_normal_500_hi, _normal_500_lo = _get_normal_band(500, _TODAY)

UA_TEMP_BANDS_850 = _make_ua_temp_bands(850, _TODAY)
UA_TEMP_BANDS_500 = _make_ua_temp_bands(500, _TODAY)
UA_TEMP_BANDS     = UA_TEMP_BANDS_850  # backward-compat alias

UA_TEMP_BAND_BASE_850 = 0
UA_TEMP_BAND_BASE_500 = 0

_all_dates = set()
for m, d, hi, lo in _NORMALS_850: _all_dates.add((m, d))
for m, d, hi, lo in _NORMALS_500: _all_dates.add((m, d))
for s in _HEIGHT_CONTROL:
    mon, day = s.split(' ')
    _all_dates.add((_MONTHS.index(mon) + 1, int(day)))



_today_doy = _TODAY.timetuple().tm_yday
_best_active_doy = max(
    (_doy(m, d) for m, d in _all_dates if _doy(m, d) <= _today_doy),
    default=max(_doy(m, d) for m, d in _all_dates)
)


# Print normals table
_today_str = f"{_MONTHS[_TODAY.month-1]} {_TODAY.day}"
print(f'\n  Normal temperature bands  —  {_TODAY}')
print(f"  {'Date':<12} {'850 hPa':>12} {'500 hPa':>12} {'500 hPa Hgt':>13}")
print('  ' + '-' * 52)

for _m, _d in sorted(_all_dates):
    _lbl   = f"{_MONTHS[_m-1]} {_d}"
    _entry_doy = _doy(_m, _d)
    _HIGHLIGHT = '\033[103m'  # yellow background
    _RESET     = '\033[0m'
    _mark = ' ◀ active' if _entry_doy == _best_active_doy else ''
    _n850  = next(((hi,lo) for mm,dd,hi,lo in _NORMALS_850 if mm==_m and dd==_d), None)
    _n500  = next(((hi,lo) for mm,dd,hi,lo in _NORMALS_500 if mm==_m and dd==_d), None)
    _hgt   = next((v for k,v in _HEIGHT_CONTROL.items() if k==_lbl), None)
    _c850  = f"{_n850[1]}→{_n850[0]}°C" if _n850 else '—'
    _c500  = f"{_n500[1]}→{_n500[0]}°C" if _n500 else '—'
    _chgt  = f"{_hgt} m"                if _hgt  else '—'
    if _entry_doy == _best_active_doy:
      print(f"{_HIGHLIGHT}  {_lbl:<12} {_c850:>12} {_c500:>12} {_chgt:>13}{_mark}{_RESET}")
    else:
      print(f"  {_lbl:<12} {_c850:>12} {_c500:>12} {_chgt:>13}{_mark}")
print(f'\n  850 hPa normal: {_normal_850_lo} to {_normal_850_hi} °C')
print(f'  500 hPa normal: {_normal_500_lo} to {_normal_500_hi} °C')


# ══════════════════════════════════════════════════════════════════════════
#  H/L DETECTION — MetPy peak_persistence + haversine distance filter
# ══════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _find_ua_hl_metpy(hght_grid, lon_vec, lat_vec, pressure_level):
    """
    Detect upper-air H/L centres using MetPy peak_persistence.

    Returns list of dicts: {type, lat, lon, val}
    """
    smooth_n       = HL_SMOOTH_N
    min_pers       = HL_MIN_PERSISTENCE.get(pressure_level, 30.0)
    min_dist_km    = HL_MIN_DISTANCE_KM.get(pressure_level, 400.0)
    edge_skip      = HL_EDGE_SKIP_DEG

    # MetPy smooth_gaussian (GEMPAK-style Gaussian, n = crest-to-crest grid increments)
    grid_q = hght_grid * units.meter
    sm     = smooth_gaussian(grid_q, n=smooth_n).magnitude

    all_centers = []
    for typ, maxima in [('H', True), ('L', False)]:
        pp = peak_persistence(sm, maxima=maxima)
        accepted = []
        for (r, c), pers in pp:
            # persistence filter — inf always passes
            if pers != float('inf') and pers < min_pers:
                continue
            lat = float(lat_vec[r])
            lon = float(lon_vec[c])
            # edge filter
            if (lat < lat_vec[0]  + edge_skip or lat > lat_vec[-1] - edge_skip or
                lon < lon_vec[0]  + edge_skip or lon > lon_vec[-1] - edge_skip):
                continue
            # minimum-distance filter (keep highest-persistence centre)
            too_close = False
            for prev in accepted:
                if _haversine_km(lat, lon, prev['lat'], prev['lon']) < min_dist_km:
                    too_close = True
                    break
            if not too_close:
                accepted.append({
                    'type':        typ,
                    'lat':         lat,
                    'lon':         lon,
                    'val':         float(sm[r, c]),
                    'persistence': float(pers),
                })
        all_centers.extend(accepted)

    return all_centers


# ══════════════════════════════════════════════════════════════════════════
#  W/C detection
# ══════════════════════════════════════════════════════════════════════════


def _find_ua_wc_metpy(temp_grid, lon_vec, lat_vec, pressure_level):
    """
    Detect upper-air W/C thermal centres using MetPy peak_persistence
    on the temperature field. Mirrors _find_ua_hl_metpy exactly but
    operates on °C values rather than geopotential height.

    Returns list of dicts: {type, lat, lon, val, persistence}
    """
    smooth_n    = WC_SMOOTH_N
    min_pers    = WC_MIN_PERSISTENCE.get(pressure_level, 1.5)
    min_dist_km = WC_MIN_DISTANCE_KM.get(pressure_level, 300.0)
    edge_skip   = WC_EDGE_SKIP_DEG

    # smooth_gaussian expects a pint Quantity; use kelvin as unit
    # (values are °C but only the smoothing shape matters)
    grid_q = temp_grid * units.kelvin
    sm     = smooth_gaussian(grid_q, n=smooth_n).magnitude

    all_centers = []
    for typ, maxima in [('W', True), ('C', False)]:
        pp = peak_persistence(sm, maxima=maxima)
        accepted = []
        for (r, c), pers in pp:
            if pers != float('inf') and pers < min_pers:
                continue
            lat = float(lat_vec[r])
            lon = float(lon_vec[c])
            if (lat < lat_vec[0]  + edge_skip or lat > lat_vec[-1] - edge_skip or
                lon < lon_vec[0]  + edge_skip or lon > lon_vec[-1] - edge_skip):
                continue
            too_close = False
            for prev in accepted:
                if _haversine_km(lat, lon, prev['lat'], prev['lon']) < min_dist_km:
                    too_close = True
                    break
            if not too_close:
                accepted.append({
                    'type':        typ,
                    'lat':         lat,
                    'lon':         lon,
                    'val':         float(sm[r, c]),
                    'persistence': float(pers),
                })
        all_centers.extend(accepted)

    return all_centers


# ══════════════════════════════════════════════════════════════════════════
#  CONTOUR / FILL HELPERS
# ══════════════════════════════════════════════════════════════════════════

# How temp band get filled

def _build_temp_band_fills(grid, lon_vec, lat_vec, ua_temp_bands, tb_base):
    bands = [(round(min(b[0], b[1]), 1), round(max(b[0], b[1]), 1), b[2])
             for b in ua_temp_bands]
    bands = [(lo + tb_base, hi + tb_base, col) for lo, hi, col in bands]
    bands.sort(key=lambda x: x[0])
    colored_bands = bands
    if not colored_bands:
        return []

    all_edges = sorted(
        {lo for lo, hi, col in colored_bands} |
        {hi for lo, hi, col in colored_bands}
    )
    dmin, dmax = float(grid.min()), float(grid.max())
    MARGIN = 0.01
    active_edges = sorted(set(
        [dmin - MARGIN]
        + [e for e in all_edges]
        + [dmax + MARGIN]
    ))
# Still need at least one interior band boundary
    if len(active_edges) < 2:
        return []

    glon, glat = np.meshgrid(lon_vec, lat_vec)
    fig, ax = plt.subplots(figsize=(1, 1))
    all_fills = []

    try:
        cf = ax.contourf(glon, glat, grid, levels=active_edges)

        from shapely.geometry import Polygon as _SP, MultiPolygon as _MP
        from shapely.ops import unary_union as _UU

        def _segs_to_shapely(segs):
            """Convert matplotlib contourf segments to Shapely geometry.
            Uses spatial containment to identify holes vs outer fills —
            winding order is unreliable across matplotlib versions."""
            if not segs:
                return None

            polys = []
            for seg in segs:
                if len(seg) < 3:
                    continue
                try:
                    p = _SP(seg)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.is_valid and not p.is_empty:
                        polys.append(p)
                except Exception:
                    continue

            if not polys:
                return None

            # Sort largest→smallest so containment tests run large-parent-first
            polys.sort(key=lambda p: p.area, reverse=True)

            # Build containment tree using winding order as primary signal,
            # containment only as tiebreaker for ambiguous cases.
            # In contourf: CCW = filled outer, CW = hole punched into parent.
            # Containment alone fails for siblings that share a bounding region.
            n = len(polys)
            depth = [0] * n
            for i in range(1, n):
                # Primary: use winding order
                try:
                    if not polys[i].exterior.is_ccw:
                        # CW = hole — find its containing parent
                        for j in range(i - 1, -1, -1):
                            try:
                                pt = polys[i].representative_point()
                                if polys[j].contains(pt) or polys[j].covers(pt):
                                    depth[i] = depth[j] + 1
                                    break
                            except Exception:
                                continue
                        else:
                            # No parent found — treat as outer despite CW winding
                            depth[i] = 0
                        continue
                except Exception:
                    pass
                # CCW = outer — depth stays 0 unless it is inside another CCW outer
                # (i.e. an island inside a hole — depth 2). Check containment.
                for j in range(i - 1, -1, -1):
                    try:
                        pt = polys[i].representative_point()
                        if polys[j].contains(pt) or polys[j].covers(pt):
                            depth[i] = depth[j] + 1
                            break
                    except Exception:
                        continue

            # Even depth = outer filled region, odd depth = hole
            outers = [polys[i] for i in range(n) if depth[i] % 2 == 0]
            holes  = [polys[i] for i in range(n) if depth[i] % 2 == 1]

            if not outers:
                return None

            try:
                outer_union = _UU(outers) if len(outers) > 1 else outers[0]
                if not outer_union.is_valid:
                    outer_union = outer_union.buffer(0)
                if holes:
                    hole_union = _UU(holes) if len(holes) > 1 else holes[0]
                    if not hole_union.is_valid:
                        hole_union = hole_union.buffer(0)
                    result = outer_union.difference(hole_union)
                else:
                    result = outer_union
                if not result.is_valid:
                    result = result.buffer(0)
                return result
            except Exception:
                return None

        # Build cumulative geometry per contourf band index
        cumulative = {}
        for si in range(len(active_edges) - 1):
            if si >= len(cf.allsegs) or not cf.allsegs[si]:
                continue
            geom = _segs_to_shapely(cf.allsegs[si])
            if geom is not None and not geom.is_empty:
                cumulative[si] = geom

        # Subtract lower cumulative area from each band to get the ring for that band
        sorted_sis = sorted(cumulative.keys())

        for idx, si in enumerate(sorted_sis):
            lo_edge = active_edges[si]
            hi_edge = active_edges[si + 1]
            interval_mid = (lo_edge + hi_edge) / 2.0

            col = None
            for blo, bhi, bcol in colored_bands:
                if blo <= interval_mid <= bhi:
                    col = bcol
                    break
            if col is None:
                continue

            ring = cumulative[si]
            if idx > 0:
                prev_si = sorted_sis[idx - 1]
                try:
                    ring = ring.difference(cumulative[prev_si])
                    if not ring.is_valid:
                        ring = ring.buffer(0)
                except Exception:
                    pass



            if ring is None or ring.is_empty:
                continue

            geoms = list(ring.geoms) if ring.geom_type in ('MultiPolygon', 'GeometryCollection') else [ring]
            for poly in geoms:
                if poly.is_empty or poly.geom_type != 'Polygon':
                    continue
                outer = [[float(v[0]), float(v[1])] for v in poly.exterior.coords]
                holes_out = [
                    [[float(v[0]), float(v[1])] for v in interior.coords]
                    for interior in poly.interiors
                ]
                all_fills.append({
                    'color':  col,
                    'coords': outer,
                    'holes':  holes_out,
                })

        # Rebuild white fills by subtracting all colored fills from them
        _colored_polys   = []
        _colored_entries = []
        _white_entries   = []

        for _f in all_fills:
            if _f['color'] != '#ffffff':
                try:
                    _p = _SP(_f['coords'])
                    if not _p.is_valid:
                        _p = _p.buffer(0)
                    if _p.is_valid and not _p.is_empty:
                        _colored_polys.append(_p)
                except Exception:
                    pass
                _colored_entries.append(_f)
            else:
                _white_entries.append(_f)

        if _colored_polys:
            _all_colored = _UU(_colored_polys)
            _new_white   = []
            for _f in _white_entries:
                try:
                    _wp = _SP(_f['coords'])
                    if not _wp.is_valid:
                        _wp = _wp.buffer(0)
                    _wp = _wp.difference(_all_colored)
                    if _wp.is_empty:
                        continue
                    _geoms = list(_wp.geoms) if _wp.geom_type == 'MultiPolygon' else [_wp]
                    for _wg in _geoms:
                        if _wg.is_empty or _wg.geom_type != 'Polygon':
                            continue
                        _outer = [[float(v[0]), float(v[1])] for v in _wg.exterior.coords]
                        _holes = [
                            [[float(v[0]), float(v[1])] for v in _i.coords]
                            for _i in _wg.interiors
                        ]
                        _new_white.append({'color': '#ffffff', 'coords': _outer, 'holes': _holes})
                except Exception:
                    _new_white.append(_f)
            all_fills = _new_white + _colored_entries
        else:
            all_fills = _white_entries + _colored_entries

    except Exception as e:
        print(f'    ⚠ band fill error: {e}')

    plt.close(fig)
    return all_fills




def _build_contours_for_field(recs_df, field_col, interval, lvl_label,
                               fixed_levels=None, sigma=None, rbf_smooth=None):
    if sigma is None:     raise ValueError(f'sigma required for {field_col}')
    if rbf_smooth is None: raise ValueError(f'rbf_smooth required for {field_col}')
    pts = []
    for _, row in recs_df.iterrows():
        v = row.get(field_col)
        if v is None or (isinstance(v, float) and np.isnan(v)): continue
        pts.append((row['lat'], row['lon'], float(v)))
    if len(pts) < 8: return []
    seen = {}
    for la, lo, v in pts:
        seen.setdefault((round(la, 2), round(lo, 2)), []).append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in seen.items()]
    if len(pts) < 8: return []
    lats = np.array([p[0] for p in pts])
    lons = np.array([p[1] for p in pts])
    vals = np.array([p[2] for p in pts])
    pad = 1.5; N = 180
    lon_vec = np.linspace(lons.min() - pad, lons.max() + pad, N)
    lat_vec = np.linspace(lats.min() - pad, lats.max() + pad, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    try:
        rbf  = RBFInterpolator(np.column_stack([lons, lats]), vals,
                               kernel='thin_plate_spline',
                               smoothing=max(rbf_smooth * len(pts), 1e-6))
        grid = rbf(np.column_stack([glon.ravel(), glat.ravel()])).reshape(N, N)
    except Exception:
        return []
    # Use MetPy smooth_gaussian instead of scipy gaussian_filter
    grid_q  = grid * units.meter
    grid    = smooth_gaussian(grid_q, n=max(2, int(sigma * 2))).magnitude

    if fixed_levels is not None:
        levels = fixed_levels
    else:
        vmin = np.floor(grid.min() / interval) * interval
        vmax = np.ceil(grid.max()  / interval) * interval
        levels = np.arange(vmin, vmax + interval, interval)
    if len(levels) < 2: return []
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cs = ax.contour(glon, glat, grid, levels=levels)
    except Exception:
        plt.close(fig); return []
    plt.close(fig)
    segments = []
    for li, lv in enumerate(cs.levels):
        for coords in cs.allsegs[li]:
            if len(coords) < 2: continue
            mid = coords[len(coords) // 2]
            segments.append({'level':     float(lv),
                             'coords':    [[float(c[0]), float(c[1])] for c in coords],
                             'label_lon': float(mid[0]),
                             'label_lat': float(mid[1])})
    return segments


# ── Diagnostic ────────────────────────────────────────────────────────────
_diag_hr = sorted(ua_summary_df['hour'].unique())[0]
_diag_df = ua_summary_df[ua_summary_df['hour'] == _diag_hr].copy()
_pts_d   = [(float(_row['lat']), float(_row['lon']), float(_row['TEMP_850']))
             for _, _row in _diag_df.iterrows()
             if _row.get('TEMP_850') is not None
             and not (isinstance(_row['TEMP_850'], float) and np.isnan(_row['TEMP_850']))]
print(f'\n  Diag: {len(_pts_d)} temp points at 850 hPa')
if len(_pts_d) >= 8:
    _lats_d = np.array([p[0] for p in _pts_d])
    _lons_d = np.array([p[1] for p in _pts_d])
    _vals_d = np.array([p[2] for p in _pts_d])
    print(f'  Temp range: {_vals_d.min():.1f} to {_vals_d.max():.1f} °C')
    _lv_d  = np.linspace(_lons_d.min() - 1.5, _lons_d.max() + 1.5, 180)
    _ltv_d = np.linspace(_lats_d.min() - 1.5, _lats_d.max() + 1.5, 180)
    _glon_d, _glat_d = np.meshgrid(_lv_d, _ltv_d)
    _rbf_d  = RBFInterpolator(np.column_stack([_lons_d, _lats_d]), _vals_d,
                               kernel='thin_plate_spline',
                               smoothing=max(_RBF_SMOOTH['TEMP'] * len(_pts_d), 1e-6))
    _grid_d = _rbf_d(np.column_stack([_glon_d.ravel(), _glat_d.ravel()])).reshape(180, 180)
    _grid_d = smooth_gaussian(_grid_d * units.meter, n=max(2, int(_SIGMA['TEMP'] * 2))).magnitude
    print(f'  Grid range: {_grid_d.min():.1f} to {_grid_d.max():.1f} °C')
    _fills_d = _build_temp_band_fills(_grid_d, _lv_d, _ltv_d, UA_TEMP_BANDS_850, 0.0)
    print(f'  Fill polygons: {len(_fills_d)}')


# ══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — per-hour upper-air data
# ══════════════════════════════════════════════════════════════════════════

_ua_hours_available = sorted(ua_summary_df['hour'].unique())
_ts_ua = {}

for _hr in _ua_hours_available:
    _df_hr = ua_summary_df[ua_summary_df['hour'] == _hr].copy()
    _valid = _df_hr['valid_time'].iloc[0] if len(_df_hr) else f'{_hr:02d}Z'
    print(f'\n  Processing {_valid} ({len(_df_hr)} stations)...')

    _hr_data = {}

    for _plvl in [850, 700, 500, 250]:
        _lvl_data = {}

        # ── Height contours ───────────────────────────────────────────────
        _fixed = UA_HGHT_LEVELS.get(_plvl)
        _segs  = (_build_contours_for_field(_df_hr, f'HGHT_{_plvl}',
                  _INTERVALS['HGHT'], _plvl, fixed_levels=_fixed,
                  sigma=_SIGMA['HGHT'], rbf_smooth=_RBF_SMOOTH['HGHT'])
                  if _fixed is not None else [])
        _lvl_data['hght'] = _segs
        print(f'    {_plvl} hPa  HGHT: {len(_segs)} segs', end='')


        # ── Shared temperature grid (contours + fills use identical interpolation) ──
        _pts_t = [(float(_row['lat']), float(_row['lon']), float(_row[f'TEMP_{_plvl}']))
                  for _, _row in _df_hr.iterrows()
                  if _row.get(f'TEMP_{_plvl}') is not None
                  and not (isinstance(_row[f'TEMP_{_plvl}'], float)
                           and np.isnan(_row[f'TEMP_{_plvl}']))]
        _shared_temp_grid = None
        _shared_lv_t = None
        _shared_ltv_t = None
        if len(_pts_t) >= 8:
            _seen_t = {}
            for _la, _lo, _v in _pts_t:
                _seen_t.setdefault((round(_la, 2), round(_lo, 2)), []).append(_v)
            _pts_t = [(_k[0], _k[1], float(np.mean(_vs))) for _k, _vs in _seen_t.items()]
            _lats_t = np.array([p[0] for p in _pts_t])
            _lons_t = np.array([p[1] for p in _pts_t])
            _vals_t = np.array([p[2] for p in _pts_t])
            _pad_t = 1.5; _NT = 180
            _shared_lv_t  = np.linspace(_lons_t.min() - _pad_t, _lons_t.max() + _pad_t, _NT)
            _shared_ltv_t = np.linspace(_lats_t.min() - _pad_t, _lats_t.max() + _pad_t, _NT)
            _glon_t, _glat_t = np.meshgrid(_shared_lv_t, _shared_ltv_t)
            try:
                _rbf_t = RBFInterpolator(
                    np.column_stack([_lons_t, _lats_t]), _vals_t,
                    kernel='thin_plate_spline',
                    smoothing=max(_RBF_SMOOTH['TEMP'] * len(_pts_t), 1e-6))
                _shared_temp_grid = _rbf_t(
                    np.column_stack([_glon_t.ravel(), _glat_t.ravel()])).reshape(_NT, _NT)
                _shared_temp_grid = smooth_gaussian(
                    _shared_temp_grid * units.meter,
                    n=max(2, int(_SIGMA['TEMP'] * 2))).magnitude
            except Exception as _et:
                print(f'  TEMP grid failed ({_et})', end='')

        # ── Temperature contours — from shared grid ───────────────────────
        _segs = []
        if _shared_temp_grid is not None:
            _vmin_t = np.floor(_shared_temp_grid.min() / _INTERVALS['TEMP']) * _INTERVALS['TEMP']
            _vmax_t = np.ceil(_shared_temp_grid.max()  / _INTERVALS['TEMP']) * _INTERVALS['TEMP']
            _t_levels = np.arange(_vmin_t, _vmax_t + _INTERVALS['TEMP'], _INTERVALS['TEMP'])
            if len(_t_levels) >= 2:
                _fig_tc, _ax_tc = plt.subplots(figsize=(1, 1))
                try:
                    _glon_t2, _glat_t2 = np.meshgrid(_shared_lv_t, _shared_ltv_t)
                    _cs_t = _ax_tc.contour(_glon_t2, _glat_t2, _shared_temp_grid,
                                           levels=_t_levels)
                    for _li, _lv in enumerate(_cs_t.levels):
                        for _coords in _cs_t.allsegs[_li]:
                            if len(_coords) < 2: continue
                            _mid = _coords[len(_coords) // 2]
                            _segs.append({'level':     float(_lv),
                                          'coords':    [[float(c[0]), float(c[1])]
                                                        for c in _coords],
                                          'label_lon': float(_mid[0]),
                                          'label_lat': float(_mid[1])})
                except Exception:
                    pass
                plt.close(_fig_tc)

        # ── W/C thermal centres — from same shared temp grid ─────────────
        _wc_centres = []
        if _shared_temp_grid is not None and _plvl in WC_LEVELS:
            try:
                _wc_centres = _find_ua_wc_metpy(
                    _shared_temp_grid, _shared_lv_t, _shared_ltv_t, _plvl)
                _nw = sum(1 for x in _wc_centres if x['type'] == 'W')
                _nc = sum(1 for x in _wc_centres if x['type'] == 'C')
                print(f'  W/C {_plvl} hPa: {_nw} Warm, {_nc} Cold', end='')
                for _c in _wc_centres:
                    print(f'\n    {_c["type"]}  lat={_c["lat"]:.1f}  '
                          f'lon={_c["lon"]:.1f}  val={_c["val"]:.1f}°C  '
                          f'pers={_c["persistence"]:.2f}')
            except Exception as _ew:
                print(f'  W/C failed ({_ew})', end='')
        _lvl_data['wc'] = _wc_centres
        _lvl_data['temp'] = _segs
        print(f'  TEMP: {len(_segs)}', end='')

        # ── Temperature band fills — from same shared grid ────────────────
        _band_fills = []
        if _shared_temp_grid is not None:
            try:
                _bands_for_lvl = (UA_TEMP_BANDS_500 if _plvl in [500, 250]
                                  else UA_TEMP_BANDS_850)
                _band_fills = _build_temp_band_fills(
                    _shared_temp_grid, _shared_lv_t, _shared_ltv_t,
                    _bands_for_lvl, 0.0)
                print(f'  fills: {len(_band_fills)}', end='')
            except Exception as _ef:
                print(f'  fills: failed ({_ef})', end='')
        _lvl_data['temp_band_fills'] = _band_fills



        # ── T-Td contours ─────────────────────────────────────────────────
        _df_ttd = _df_hr.copy()
        _t_col, _d_col = f'TEMP_{_plvl}', f'DWPT_{_plvl}'
        if _t_col in _df_ttd.columns and _d_col in _df_ttd.columns:
            _df_ttd[f'TTDP_{_plvl}'] = _df_ttd[_t_col] - _df_ttd[_d_col]
            _segs = _build_contours_for_field(
                _df_ttd, f'TTDP_{_plvl}', _INTERVALS['TTDP'], _plvl,
                sigma=_SIGMA['TTDP'], rbf_smooth=_RBF_SMOOTH['TTDP'])
        else:
            _segs = []
        _lvl_data['ttdp'] = _segs
        print(f'  T-Td: {len(_segs)}', end='')

        # ── Wind speed contours ───────────────────────────────────────────
        _segs = _build_contours_for_field(_df_hr, f'SPED_{_plvl}', _INTERVALS['SPED'], _plvl,
                  sigma=_SIGMA['SPED'], rbf_smooth=_RBF_SMOOTH['SPED'])
        _lvl_data['sped'] = _segs
        print(f'  WIND: {len(_segs)}')

        _hr_data[str(_plvl)] = _lvl_data

    # ── Instability contours (T700 - T500) ───────────────────────────────
    _instab_cts = []
    _stn_lats, _stn_lons, _stn_vals = [], [], []
    for _, _row in _df_hr.iterrows():
        _v5 = _row.get('TEMP_500'); _v7 = _row.get('TEMP_700')
        if (_v5 is None or _v7 is None
                or (isinstance(_v5, float) and np.isnan(_v5))
                or (isinstance(_v7, float) and np.isnan(_v7))): continue
        _stn_lats.append(float(_row['lat'])); _stn_lons.append(float(_row['lon']))
        _stn_vals.append(float(_v7) - float(_v5))
    _stn_lats = np.array(_stn_lats)
    _stn_lons = np.array(_stn_lons)
    _stn_vals = np.array(_stn_vals)
    if len(_stn_vals) >= 1:
        _pad = 1.5; _NI = 180
        _ilon = np.linspace(_stn_lons.min() - _pad, _stn_lons.max() + _pad, _NI)
        _ilat = np.linspace(_stn_lats.min() - _pad, _stn_lats.max() + _pad, _NI)
        _iglon, _iglat = np.meshgrid(_ilon, _ilat)
        _tree = cKDTree(np.column_stack([_stn_lons, _stn_lats]))
        _dists, _idxs = _tree.query(
            np.column_stack([_iglon.ravel(), _iglat.ravel()]), k=1)
        _nn_vals = _stn_vals[_idxs].astype(float)
        _nn_vals[_dists > 3.5] = np.nan
        _diff_grid    = _nn_vals.reshape(_NI, _NI)
        _diff_grid_sm = smooth_gaussian(
            np.where(np.isnan(_diff_grid), 0, _diff_grid) * units.kelvin,
            n=max(2, int(sigmaT700500 * 2))).magnitude
        _diff_grid_sm[np.isnan(_diff_grid)] = np.nan
        for _band_lvl, _mask_val in [(16, 16.0), (18, 18.0)]:
            _binary = np.where(
                (~np.isnan(_diff_grid_sm)) & (_diff_grid_sm >= _band_lvl) &
                (_diff_grid_sm < (_band_lvl + 2)
                 if _band_lvl == 16 else np.ones_like(_diff_grid_sm, bool)),
                1.0, 0.0)
            if _binary.max() < 0.5: continue
            _fig_i, _ax_i = plt.subplots(figsize=(1, 1))
            try:
                _cs_i = _ax_i.contour(_iglon, _iglat, _binary, levels=[0.5])
                for _seg in _cs_i.allsegs[0]:
                    if len(_seg) < 3: continue
                    _mid_i = _seg[len(_seg) // 2]
                    _instab_cts.append({
                        'level':     float(_mask_val),
                        'coords':    [[float(p[0]), float(p[1])] for p in _seg],
                        'label_lon': float(_mid_i[0]),
                        'label_lat': float(_mid_i[1]),
                    })
            except Exception:
                pass
            plt.close(_fig_i)
    print(f'    Instability: {len(_instab_cts)} segs')

    # ── H/L detection — MetPy peak_persistence ───────────────────────────
    _ua_hl_all = {}
    for _plvl in HL_LEVELS:
        _hght_col = f'HGHT_{_plvl}'
        _pts = [(float(_row['lat']), float(_row['lon']), float(_row[_hght_col]))
                for _, _row in _df_hr.iterrows()
                if _row.get(_hght_col) is not None
                and not (isinstance(_row[_hght_col], float) and np.isnan(_row[_hght_col]))]
        if len(_pts) < 8:
            print(f'  {_plvl} hPa H/L: insufficient data ({len(_pts)} pts), skipped')
            _ua_hl_all[_plvl] = []
            continue
        # Deduplicate overlapping stations
        _seen = {}
        for _la, _lo, _v in _pts:
            _seen.setdefault((round(_la, 2), round(_lo, 2)), []).append(_v)
        _pts = [(_k[0], _k[1], float(np.mean(_vs))) for _k, _vs in _seen.items()]
        _lats_u = np.array([p[0] for p in _pts])
        _lons_u = np.array([p[1] for p in _pts])
        _vals_u = np.array([p[2] for p in _pts])
        _pad = 1.5; _NU = 180
        _lv_u  = np.linspace(_lons_u.min() - _pad, _lons_u.max() + _pad, _NU)
        _ltv_u = np.linspace(_lats_u.min() - _pad, _lats_u.max() + _pad, _NU)
        _glon_u, _glat_u = np.meshgrid(_lv_u, _ltv_u)
        try:
            _rbf_u = RBFInterpolator(np.column_stack([_lons_u, _lats_u]), _vals_u,
                                     kernel='thin_plate_spline',
                                     smoothing=max(0.3 * len(_pts), 1e-6))
            _hght_grid = _rbf_u(
                np.column_stack([_glon_u.ravel(), _glat_u.ravel()])).reshape(_NU, _NU)
        except Exception as _e:
            print(f'  {_plvl} hPa H/L: RBF failed ({_e}), skipped')
            _ua_hl_all[_plvl] = []
            continue

        _ua_hl = _find_ua_hl_metpy(_hght_grid, _lv_u, _ltv_u, _plvl)
        _ua_hl_all[_plvl] = _ua_hl
        _nh = sum(1 for x in _ua_hl if x['type'] == 'H')
        _nl = sum(1 for x in _ua_hl if x['type'] == 'L')
        print(f'  {_plvl} hPa H/L @ {_hr:02d}Z: {_nh} High(s), {_nl} Low(s)')
        for _c in _ua_hl:
            print(f'    {_c["type"]}  lat={_c["lat"]:.1f}  lon={_c["lon"]:.1f}'
                  f'  val={_c["val"]:.0f}m  pers={_c["persistence"]:.1f}')

    # ── Thermal ridge/trough from earlier blocks ──────────────────────────
    def _seg_to_dict(seg):
        mid = seg[len(seg) // 2]
        return {'coords':    [[float(p[1]), float(p[0])] for p in seg],
                'label_lon': float(mid[1]),
                'label_lat': float(mid[0])}

    _ts_ua[str(int(_hr))] = {
        'levels':             _hr_data,
        'instab':             _instab_cts,
        'thermal_ridge_850':  [_seg_to_dict(s) for s in globals().get(f'ridge_segs_{int(_hr):02d}',      [])],
        'thermal_trough_850': [_seg_to_dict(s) for s in globals().get(f'trough_segs_{int(_hr):02d}',     [])],
        'thermal_ridge_700':  [_seg_to_dict(s) for s in globals().get(f'ridge_segs_700_{int(_hr):02d}',  [])],
        'thermal_trough_700': [_seg_to_dict(s) for s in globals().get(f'trough_segs_700_{int(_hr):02d}', [])],
        'thermal_ridge_500':  [_seg_to_dict(s) for s in globals().get(f'ridge_segs_500_{int(_hr):02d}',  [])],
        'thermal_trough_500': [_seg_to_dict(s) for s in globals().get(f'trough_segs_500_{int(_hr):02d}', [])],
        'dtdx_zero_pts':      [],
        **{f'hl_{pl}': _ua_hl_all.get(pl, []) for pl in HL_LEVELS},

        # ── new ────W/C──────────────────────────────────────────────────────
        **{f'wc_{pl}': (
               [c for lvl_d in [_hr_data.get(str(pl), {})]
                  for c in lvl_d.get('wc', [])]
           ) for pl in WC_LEVELS},

    }
    print(f'    → stored hour {int(_hr)} with {len(_instab_cts)} instab segs')


# ── H/L diagnostic ────────────────────────────────────────────────────────
print('\n  H/L diagnostic:')
for _hr_check in _ua_hours_available:
    _k = str(int(_hr_check))
    for _pl in HL_LEVELS:
        _hl_list = _ts_ua[_k].get(f'hl_{_pl}', 'KEY MISSING')
        print(f'    hr={_k} {_pl}hPa: {_hl_list}')

_ts_ua_json_str = _json_ua.dumps(_ts_ua)
print(f'\n✅ Block 05 complete — {len(_ts_ua)} hour(s), {len(_ts_ua_json_str)//1024} KB')


# ── Band scale visualisation ──────────────────────────────────────────────
def _plot_band_legend(ax, bands, normal_hi, normal_lo, vmin=-60, vmax=60, title=''):
    ax.set_xlim(vmin, vmax); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=6)
    for (hi, lo, col) in bands:
        lo_c, hi_c = max(lo, vmin), min(hi, vmax)
        if hi_c <= lo_c: continue
        ax.add_patch(_mpatches.FancyBboxPatch(
            (lo_c, 0), hi_c - lo_c, 1,
            boxstyle='square,pad=0', facecolor=col,
            edgecolor='#888888', linewidth=0.4))
    boundaries = ({lo for _,lo,_ in bands if vmin<=lo<=vmax}
                  | {hi for hi,_,_ in bands if vmin<=hi<=vmax})
    for x in sorted(boundaries):
        ax.axvline(x, color='#666666', linewidth=0.5, alpha=0.7, zorder=4)
        ax.text(x, -0.18, str(int(x)), ha='center', va='top', fontsize=7,
                color='#333333', transform=ax.get_xaxis_transform(), clip_on=False)
    mid = (normal_lo + normal_hi) / 2
    ax.axvline(mid, color='black', linewidth=1.2, linestyle='--', alpha=0.65, zorder=5)
    ax.annotate(f'Normal  {normal_lo}→{normal_hi} °C', xy=(mid, 1),
                xytext=(mid, 1.38), fontsize=8, ha='center', color='#333333',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
                annotation_clip=False)
    ax.set_xticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

_fig_demo, _axes_demo = plt.subplots(2, 1, figsize=(14, 4),
                                      gridspec_kw={'hspace': 1.0})
_fig_demo.suptitle(f'UA Temperature Band Scales — {_TODAY.strftime("%d %b %Y")}',
                   fontsize=12, y=1.04)
_plot_band_legend(_axes_demo[0], UA_TEMP_BANDS_850, _normal_850_hi, _normal_850_lo,
                  title=f'850 hPa  (normal {_normal_850_lo}→{_normal_850_hi} °C = green)')
_plot_band_legend(_axes_demo[1], UA_TEMP_BANDS_500, _normal_500_hi, _normal_500_lo,
                  title=f'500 hPa  (normal {_normal_500_lo}→{_normal_500_hi} °C = green)')
_buf = _io.BytesIO()
_fig_demo.savefig(_buf, format='png', dpi=130, bbox_inches='tight')
plt.close(_fig_demo)
_buf.seek(0)
# (preview skipped in headless mode)


print('\n  W/C diagnostic:')
for _hr_check in _ua_hours_available:
    _k = str(int(_hr_check))
    for _pl in WC_LEVELS:
        _wc_list = _ts_ua[_k].get(f'wc_{_pl}', 'KEY MISSING')
        print(f'    hr={_k} {_pl}hPa: {_wc_list}')

# Commented out IPython magic to ensure Python compatibility.
# ── Diagnostic plot: 4-panel (850 / 700 / 500 / 250 hPa) ─────────────────

SKIP = True

try:
    if SKIP:
        raise SystemExit()

    # Your actual code below
    print("Diagnostic plot - Skipped")

except SystemExit:
    pass  # silently skip, no output at all
else:
    pass  # only reaches here if SKIP is False and no exception

if not SKIP:
    pass
#     %matplotlib inline
    import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

_DIAG_LON = (-145, -88)
_DIAG_LAT = (28,    77)
_DIAG_HR  = str(int(_ua_hours_available[-1]))  # change to [0] for 00Z
_LEVELS   = ['850', '700', '500', '250']

_CMAP = LinearSegmentedColormap.from_list('ua_temp', [
    '#4B0082','#0000FF','#0080FF','#00BFFF','#40E0D0',
    '#7FFF00','#FFFF00','#FFA500','#FF4500','#FF0000'
])

_hr_df = ua_summary_df[ua_summary_df['hour'] == float(_DIAG_HR)].copy()
_dom_cx = sum(_DIAG_LON) / 2
_dom_cy = sum(_DIAG_LAT) / 2


def _make_grid(df, col, NB=220):
    pts = df[df[col].notna()][['lat', 'lon', col]].values
    if len(pts) < 6: return None, None, None, None, None
    lats, lons, vals = pts[:,0], pts[:,1], pts[:,2]
    pad = 2.0
    lv  = np.linspace(max(lons.min()-pad, -180), min(lons.max()+pad, 0), NB)
    ltv = np.linspace(lats.min()-pad, lats.max()+pad, NB)
    glon, glat = np.meshgrid(lv, ltv)
    rbf  = RBFInterpolator(
        np.column_stack([lons, lats]), vals,
        kernel='thin_plate_spline',
        smoothing=max(_RBF_SMOOTH[col.split('_')[0]] * len(pts), 1e-6))
    grid = rbf(np.column_stack([glon.ravel(), glat.ravel()])).reshape(NB, NB)
    grid = gaussian_filter(grid, sigma=_SIGMA[col.split('_')[0]])
    return grid, lv, ltv, vals.min(), vals.max()


def _draw_segments(ax, entry, lw_major, lw_minor, col_major, col_minor,
                   mod_major, label_every, label_fmt, label_col, linestyle='-'):
    for seg in entry:
        coords = np.array(seg['coords'])
        if coords.ndim != 2 or coords.shape[1] < 2: continue
        lons_s, lats_s = coords[:,0], coords[:,1]
        if lons_s.max() < _DIAG_LON[0] or lons_s.min() > _DIAG_LON[1]: continue
        if lats_s.max() < _DIAG_LAT[0] or lats_s.min() > _DIAG_LAT[1]: continue
        lv = seg['level']
        is_major = (lv % mod_major == 0)
        lw    = lw_major if is_major else lw_minor
        color = col_major if is_major else col_minor
        line, = ax.plot(lons_s, lats_s, color=color, linewidth=lw,
                        alpha=0.9 if is_major else 0.55,
                        linestyle=linestyle, zorder=3)
        line.set_path_effects([
            pe.Stroke(linewidth=lw+2, foreground='#0d1117', alpha=0.5),
            pe.Normal()
        ])
        if lv % label_every == 0:
            _mask = (
                (coords[:,0] > _DIAG_LON[0]) & (coords[:,0] < _DIAG_LON[1]) &
                (coords[:,1] > _DIAG_LAT[0]) & (coords[:,1] < _DIAG_LAT[1])
            )
            _inside = coords[_mask]
            if len(_inside) == 0: continue
            _dists = (_inside[:,0]-_dom_cx)**2 + (_inside[:,1]-_dom_cy)**2
            _best  = _inside[np.argmin(_dists)]
            ax.text(float(_best[0]), float(_best[1]),
                    label_fmt.format(lv),
                    fontsize=7, color=label_col, ha='center', va='center',
                    fontfamily='monospace', fontweight='bold', zorder=8,
                    path_effects=[pe.withStroke(linewidth=3, foreground='#0d1117')])


# ── figure: 2×2 grid ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(f'Upper Air  ·  {_DIAG_HR}Z  ·  Temp fill + Height contours',
             color='#c9d1d9', fontsize=14, fontfamily='monospace', y=0.99)

for ax, _DIAG_LVL in zip(axes.flat, _LEVELS):
    _tcol = f'TEMP_{_DIAG_LVL}'
    _hcol = f'HGHT_{_DIAG_LVL}'

    ax.set_facecolor('#0d1117')
    ax.set_xlim(*_DIAG_LON)
    ax.set_ylim(*_DIAG_LAT)
    ax.set_title(f'{_DIAG_LVL} hPa', color='#c9d1d9',
                 fontsize=12, fontfamily='monospace', pad=6)
    ax.set_xlabel('Lon', color='#8b949e', fontsize=8)
    ax.set_ylabel('Lat', color='#8b949e', fontsize=8)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, linestyle='--', linewidth=0.3, color='#30363d', alpha=0.4, zorder=0)

    # temp fill
    _grid_t, _lv_t, _ltv_t, _tmin, _tmax = _make_grid(_hr_df, _tcol)
    if _grid_t is not None:
        _glon_t, _glat_t = np.meshgrid(_lv_t, _ltv_t)
        cf = ax.contourf(_glon_t, _glat_t, _grid_t,
                         levels=24, cmap=_CMAP,
                         vmin=_tmin-2, vmax=_tmax+2,
                         alpha=0.82, zorder=1)
        cb = fig.colorbar(cf, ax=ax, shrink=0.55, pad=0.02)
        cb.set_label('°C', color='#8b949e', fontsize=8)
        cb.ax.tick_params(colors='#8b949e', labelsize=7)
        cb.outline.set_edgecolor('#30363d')

    # temp isotherms
    _temp_segs = (_ts_ua.get(_DIAG_HR, {})
                        .get('levels', {})
                        .get(_DIAG_LVL, {})
                        .get('temp', []))
    _draw_segments(ax, _temp_segs,
                   lw_major=1.8, lw_minor=0.7,
                   col_major='#ffffff', col_minor='#99bbcc',
                   mod_major=10, label_every=4,
                   label_fmt='{:.0f}°', label_col='#ffffff')

    # height contours
    _hght_segs = (_ts_ua.get(_DIAG_HR, {})
                        .get('levels', {})
                        .get(_DIAG_LVL, {})
                        .get('hght', []))
    _draw_segments(ax, _hght_segs,
                   lw_major=2.0, lw_minor=1.0,
                   col_major='#ffdd00', col_minor='#cc9900',
                   mod_major=120, label_every=60,
                   label_fmt='{:.0f}', label_col='#ffdd00',
                   linestyle='--')

    # ridge / trough
    for _rkey, _rcol in [(f'thermal_ridge_{_DIAG_LVL}',  '#ff44aa'),
                          (f'thermal_trough_{_DIAG_LVL}', '#44aaff')]:
        for seg in _ts_ua.get(_DIAG_HR, {}).get(_rkey, []):
            coords = np.array(seg['coords'])
            if coords.ndim != 2: continue
            ax.plot(coords[:,0], coords[:,1],
                    color=_rcol, linewidth=2.0, alpha=0.9, zorder=5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='#0d1117',
                                            alpha=0.5), pe.Normal()])

    # H/L centres
    for _hl in _ts_ua.get(_DIAG_HR, {}).get(f'hl_{_DIAG_LVL}', []):
        _sym   = _hl['type']
        _hcol2 = '#ff4444' if _sym == 'L' else '#44ff88'
        ax.text(_hl['lon'], _hl['lat'], _sym,
                fontsize=18, fontweight='black', color=_hcol2,
                ha='center', va='center', zorder=10,
                fontfamily='sans-serif',
                path_effects=[pe.withStroke(linewidth=3, foreground='#0d1117')])
        ax.text(_hl['lon'], _hl['lat'] - 1.5,
                f"{_hl['val']:.0f}m",
                fontsize=7, color=_hcol2, ha='center', zorder=10,
                fontfamily='monospace',
                path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])

    # stations
    _valid = _hr_df[
        _hr_df[_tcol].notna() &
        _hr_df['lon'].between(*_DIAG_LON) &
        _hr_df['lat'].between(*_DIAG_LAT)
    ].copy()
    if len(_valid):
        ax.scatter(_valid['lon'], _valid['lat'],
                   c=_valid[_tcol], cmap=_CMAP,
                   vmin=(_tmin-2 if _grid_t is not None else None),
                   vmax=(_tmax+2 if _grid_t is not None else None),
                   s=30, zorder=6, edgecolors='#ffffff', linewidths=0.6)
        for _, r in _valid.iterrows():
            ax.text(r['lon']+0.3, r['lat']+0.3, f"{r[_tcol]:.0f}°",
                    fontsize=7, color='#ffffff', fontweight='bold',
                    fontfamily='monospace', zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])

    print(f'  {_DIAG_LVL} hPa: temp_segs={len(_temp_segs)}'
          f'  hght_segs={len(_hght_segs)}'
          f'  H/L={len(_ts_ua.get(_DIAG_HR,{}).get(f"hl_{_DIAG_LVL}",[]))}'
          f'  stns={len(_valid)}')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print('=' * 60)
print('  BLOCK 06 — Surface & 850 hPa Convergence Zone Detection')
print('  Computes horizontal divergence from wind obs, contours at threshold')
print('=' * 60)

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json as _json_conv
import math

# ══════════════════════════════════════════════════════════
#  USER PARAMETERS
# ══════════════════════════════════════════════════════════
CONV_THRESHOLD  = -1e-5   # s⁻¹  (negative = convergence)
CONV_GRID_N     = 100
CONV_RBF_SMOOTH = 0.4
CONV_SIGMA      = 3.0
CONV_DIV_LEVELS = np.linspace(-8e-5, 8e-5, 33)
CONV_COLOR      = '#cc00cc'
CONV_FILL_COLOR = '#dd88ff'
CONV_FILL_OPACITY = 0.30
CONV_WEIGHT     = 2.5


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def _deg2m_lat(dlat_deg):
    return dlat_deg * 111_320.0

def _deg2m_lon(dlon_deg, lat_deg):
    return dlon_deg * 111_320.0 * np.cos(np.radians(lat_deg))

def compute_divergence_grid(u_grid, v_grid, lon_vec, lat_vec):
    """∂u/∂x + ∂v/∂y on a regular lat/lon grid (centred finite differences)."""
    ny, nx = u_grid.shape
    mean_lat = lat_vec
    dlon_m   = np.zeros((ny, nx))
    for i in range(ny):
        dx_deg    = lon_vec[1] - lon_vec[0]
        dlon_m[i, :] = _deg2m_lon(dx_deg, mean_lat[i])
    du_dx = np.gradient(u_grid, axis=1) / dlon_m
    dlat_m = _deg2m_lat(lat_vec[1] - lat_vec[0])
    dv_dy  = np.gradient(v_grid, axis=0) / dlat_m
    return du_dx + dv_dy


def build_wind_grids(pts_u, pts_v, N=CONV_GRID_N,
                     rbf_smooth=CONV_RBF_SMOOTH, sigma=CONV_SIGMA, pad=1.5):
    """Interpolate scattered (lat, lon, u/v) onto a regular grid."""
    if len(pts_u) < 6 or len(pts_v) < 6:
        return None, None, None, None

    def _dedup(pts):
        seen = {}
        for la, lo, v in pts:
            k = (round(la, 1), round(lo, 1))
            seen.setdefault(k, []).append(v)
        return [(k[0], k[1], float(np.mean(vs))) for k, vs in seen.items()]

    pts_u = _dedup(pts_u)
    pts_v = _dedup(pts_v)
    if len(pts_u) < 6 or len(pts_v) < 6:
        return None, None, None, None

    lats_u = np.array([p[0] for p in pts_u])
    lons_u = np.array([p[1] for p in pts_u])
    vals_u = np.array([p[2] for p in pts_u])
    lats_v = np.array([p[0] for p in pts_v])
    lons_v = np.array([p[1] for p in pts_v])
    vals_v = np.array([p[2] for p in pts_v])

    lat_min = min(lats_u.min(), lats_v.min()) - pad
    lat_max = max(lats_u.max(), lats_v.max()) + pad
    lon_min = min(lons_u.min(), lons_v.min()) - pad
    lon_max = max(lons_u.max(), lons_v.max()) + pad

    lon_vec = np.linspace(lon_min, lon_max, N)
    lat_vec = np.linspace(lat_min, lat_max, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    qi = np.column_stack([glon.ravel(), glat.ravel()])

    def _rbf(lons, lats, vals):
        try:
            rbf = RBFInterpolator(
                np.column_stack([lons, lats]), vals,
                kernel='thin_plate_spline',
                smoothing=max(rbf_smooth * len(vals), 1e-6)
            )
        except np.linalg.LinAlgError:
            rbf = RBFInterpolator(
                np.column_stack([lons, lats]), vals,
                kernel='linear',
                smoothing=max(rbf_smooth * len(vals), 1.0)
            )
        return gaussian_filter(rbf(qi).reshape(N, N), sigma=sigma)

    return _rbf(lons_u, lats_u, vals_u), _rbf(lons_v, lats_v, vals_v), lon_vec, lat_vec


def extract_convergence_contours(div_grid, lon_vec, lat_vec,
                                  threshold=CONV_THRESHOLD):
    """Contour divergence grid at threshold; return segment dicts."""
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        cs = ax.contour(glon, glat, div_grid, levels=[threshold])
    except Exception:
        plt.close(fig)
        return []
    plt.close(fig)
    segs = []
    for li, lv in enumerate(cs.levels):
        for coords in cs.allsegs[li]:
            if len(coords) < 3: continue
            mid = coords[len(coords) // 2]
            segs.append({
                'level':     float(lv),
                'coords':    [[float(c[0]), float(c[1])] for c in coords],
                'label_lon': float(mid[0]),
                'label_lat': float(mid[1]),
            })
    return segs


def _drct_sped_to_uv(drct_deg, sped_ms):
    """Met wind dir + speed → (u, v) m/s."""
    rad = math.radians(drct_deg)
    return -sped_ms * math.sin(rad), -sped_ms * math.cos(rad)


# ══════════════════════════════════════════════════════════
#  SFC CONVERGENCE  (from METAR wind obs)
# ══════════════════════════════════════════════════════════
print('\n  Computing SFC convergence...')
_ts_all_conv    = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
_conv_sfc_by_ts = {}

_ts_conv_filtered = [ts for ts in _ts_all_conv
                     if any(ts[2:6] in ('0000', '1200') for _ in [1])]
for _ts in _ts_conv_filtered:
    _recs = [d for d in metar_records if d['timestamp'] == _ts]
    pts_u, pts_v = [], []
    for d in _recs:
        _wd = d.get('wind_dir'); _ws = d.get('wind_spd')
        if _wd is None or _ws is None: continue
        try:
            _wd = float(_wd); _ws = float(_ws)
        except (TypeError, ValueError):
            continue
        if not (0 <= _wd <= 360) or not (0 <= _ws <= 100): continue
        _ws_ms = _ws * 0.514444 if _ws > 5 else _ws
        _u, _v = _drct_sped_to_uv(_wd, _ws_ms)
        pts_u.append((d['lat'], d['lon'], _u))
        pts_v.append((d['lat'], d['lon'], _v))

    if len(pts_u) < 12:
        _conv_sfc_by_ts[_ts] = []
        print(f'    SFC {_ts}: skipped ({len(pts_u)} pts < 12)')
        continue

    u_g, v_g, lv, ltv = build_wind_grids(pts_u, pts_v)
    if u_g is None:
        _conv_sfc_by_ts[_ts] = []
        print(f'    SFC {_ts}: insufficient wind data ({len(pts_u)} pts)')
        continue

    div_g = compute_divergence_grid(u_g, v_g, lv, ltv)
    segs  = extract_convergence_contours(div_g, lv, ltv)
    _conv_sfc_by_ts[_ts] = segs
    print(f'    SFC {_ts}: div [{div_g.min():.2e}, {div_g.max():.2e}]  segs={len(segs)}')
    del u_g, v_g, div_g

print(f'  ✓ SFC convergence: {len(_conv_sfc_by_ts)} timestamps')


# ══════════════════════════════════════════════════════════
#  850 hPa CONVERGENCE  (from ua_summary_df)
# ══════════════════════════════════════════════════════════
print('\n  Computing 850 hPa convergence...')
_conv_850_by_hr = {}

for _hr in sorted(ua_summary_df['hour'].unique()):
    _df = ua_summary_df[ua_summary_df['hour'] == _hr].copy()
    pts_u, pts_v = [], []
    for _, row in _df.iterrows():
        _wd = row.get('DRCT_850'); _ws = row.get('SPED_850')
        if _wd is None or _ws is None: continue
        if isinstance(_wd, float) and np.isnan(_wd): continue
        if isinstance(_ws, float) and np.isnan(_ws): continue
        _u, _v = _drct_sped_to_uv(float(_wd), float(_ws))
        pts_u.append((float(row['lat']), float(row['lon']), _u))
        pts_v.append((float(row['lat']), float(row['lon']), _v))

    if len(pts_u) < 6:
        _conv_850_by_hr[str(int(_hr))] = []
        print(f'    850 hr={int(_hr)}: skipped ({len(pts_u)} stns < 6)')
        continue

    u_g, v_g, lv, ltv = build_wind_grids(pts_u, pts_v)
    if u_g is None:
        _conv_850_by_hr[str(int(_hr))] = []
        print(f'    850 hr={int(_hr)}: insufficient data ({len(pts_u)} stns)')
        continue

    div_g = compute_divergence_grid(u_g, v_g, lv, ltv)
    segs  = extract_convergence_contours(div_g, lv, ltv)
    _conv_850_by_hr[str(int(_hr))] = segs
    print(f'    850 hr={int(_hr)}: div [{div_g.min():.2e}, {div_g.max():.2e}]  segs={len(segs)}')
    del u_g, v_g, div_g

print(f'  ✓ 850 convergence: {len(_conv_850_by_hr)} hours')


# ══════════════════════════════════════════════════════════
#  SFC TROUGH DETECTION  (from SLP)
# ══════════════════════════════════════════════════════════
print('\n  Computing SFC trough...')
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

def _build_slp_local(_recs):
    _pts = [(d['lat'], d['lon'], d['slp']) for d in _recs if d.get('slp') is not None]
    if len(_pts) < 8: return None, None, None
    _la = np.array([p[0] for p in _pts])
    _lo = np.array([p[1] for p in _pts])
    _va = np.array([p[2] for p in _pts])
    _pad = 1.5; _N = 150
    _lv  = np.linspace(_lo.min()-_pad, _lo.max()+_pad, _N)
    _ltv = np.linspace(_la.min()-_pad, _la.max()+_pad, _N)
    _GL, _GLA = np.meshgrid(_lv, _ltv)
    try:
        _rbf  = RBFInterpolator(np.column_stack([_lo, _la]), _va,
                                kernel='thin_plate_spline',
                                smoothing=max(0.3*len(_pts), 1e-6))
        _grid = _rbf(np.column_stack([_GL.ravel(), _GLA.ravel()])).reshape(_N, _N)
    except Exception:
        return None, None, None
    return gaussian_filter(_grid, sigma=3.0), _lv, _ltv

def _pts_to_segs_local(_pts, max_dist=4.0, min_pts=4):
    if len(_pts) < min_pts: return []
    _arr = np.array(_pts)
    _dedup = {}
    for _r in _arr:
        _k = (round(_r[0]*2)/2, round(_r[1]*2)/2)
        if _k not in _dedup or _r[2] > _dedup[_k][2]: _dedup[_k] = _r
    _arr = np.array(list(_dedup.values()))
    if len(_arr) < min_pts: return []
    _tree  = cKDTree(_arr[:,:2])
    _pairs = _tree.query_pairs(max_dist)
    _parent = list(range(len(_arr)))
    def _find(_x):
        while _parent[_x] != _x:
            _parent[_x] = _parent[_parent[_x]]; _x = _parent[_x]
        return _x
    for _a, _b in _pairs:
        _ra, _rb = _find(_a), _find(_b)
        if _ra != _rb: _parent[_ra] = _rb
    _clusters = {}
    for _i in range(len(_arr)):
        _clusters.setdefault(_find(_i), []).append(_i)
    _segs = []
    for _idxs in _clusters.values():
        if len(_idxs) < min_pts: continue
        _sp = sorted([(_arr[_i][0], _arr[_i][1]) for _i in _idxs], key=lambda p: p[0])
        _sub, _cur = [], [_sp[0]]
        for _k in range(1, len(_sp)):
            if (abs(_sp[_k][0]-_sp[_k-1][0]) > max_dist*1.5 or
                abs(_sp[_k][1]-_sp[_k-1][1]) > max_dist*1.5):
                if len(_cur) >= min_pts: _sub.append(_cur)
                _cur = []
            _cur.append(_sp[_k])
        if len(_cur) >= min_pts: _sub.append(_cur)
        _segs.extend(_sub)
    return _segs

_sfc_trough_by_ts = {}

for _ts in _ts_conv_filtered:
    _recs = [d for d in metar_records if d.get('timestamp') == _ts]
    _slp_grid_t, _slp_lv_t, _slp_ltv_t = _build_slp_local(_recs)
    if _slp_grid_t is None:
        _sfc_trough_by_ts[_ts] = []
        continue

    _Ts = gaussian_filter(_slp_grid_t, sigma=2.0)
    _trough_pts = []
    for _j, _lat in enumerate(_slp_ltv_t):
        _row_r = _Ts[_j, :]
        _tr, _tpr = find_peaks(-_row_r, prominence=0.3, width=2)
        if len(_tr):
            _top = np.argsort(_tpr['prominences'])[-4:]
            for _idx in _tr[_top]:
                _prom = _tpr['prominences'][np.where(_tr == _idx)[0][0]]
                _trough_pts.append((_lat, _slp_lv_t[_idx], float(_prom)))

    _segs = _pts_to_segs_local(_trough_pts, max_dist=4.0, min_pts=4)
    _out = []
    for _seg in _segs:
        _mid = _seg[len(_seg)//2]
        _out.append({
            'coords':    [[p[1], p[0]] for p in _seg],
            'label_lat': _mid[0],
            'label_lon': _mid[1],
        })
    _sfc_trough_by_ts[_ts] = _out

print(f'  ✓ SFC trough: '
      f'{sum(len(v) for v in _sfc_trough_by_ts.values())} total segs '
      f'across {len(_sfc_trough_by_ts)} timestamps')


# ══════════════════════════════════════════════════════════
#  PACKAGE INTO JSON
# ══════════════════════════════════════════════════════════
_conv_payload = {
    'sfc':        _conv_sfc_by_ts,
    '850':        _conv_850_by_hr,
    'sfc_trough': _sfc_trough_by_ts,
    'threshold':  CONV_THRESHOLD,
}
_conv_json_str = _json_conv.dumps(_conv_payload)
print(f'\n✅ Block 06 complete — convergence JSON: {len(_conv_json_str)//1024} KB')
print(f'   Variables: _conv_sfc_by_ts, _conv_850_by_hr, '
      f'_sfc_trough_by_ts, _conv_json_str')

# ── Tooltip toggle control ─────────────────────────────────────────────────
_TOOLTIP_ON = False   # ← set False to start with tooltips muted. Good for export

# Building the Chart
# -- Cell 9 - Interactive Folium map with OSM tiles ---
import folium
from folium import Element
import json as _json
import numpy as np
from matplotlib import pyplot as plt
import math as _math



# ---- 500 hPa KEY HEIGHT — auto-adjusted by date ------------------------
from datetime import date as _date_kh

_HEIGHT_CONTROL = {
    "Jan 1":  5400,
    "Apr 3":  5460,
    "Apr 19": 5520,
    "May 11": 5580,
    "May 30": 5640,
    "Jun 27": 5700,
    "Jul 26": 5760,
    "Aug 7":  5700,
    "Aug 31": 5640,
    "Oct 1":  5580,
    "Oct 17": 5520,
    "Oct 29": 5460,
    "Nov 17": 5400,
}

def _parse_height_entry(label_str, ref_year=2001):
    """Convert 'Mon D' string to day-of-year."""
    from datetime import datetime
    return datetime.strptime(f"{label_str} {ref_year}", "%b %d %Y").timetuple().tm_yday

def _get_key_hgt_500(today=None):
    """
    Step function: return the 500 hPa key height for today's date.
    Uses the most recent past entry in _HEIGHT_CONTROL.
    Wraps to the last entry if today is before the first entry.
    """
    if today is None:
        today = _date_kh.today()
    today_doy = today.timetuple().tm_yday

    best_val = None
    best_doy = -1
    for label_str, hgt in _HEIGHT_CONTROL.items():
        entry_doy = _parse_height_entry(label_str)
        if entry_doy <= today_doy and entry_doy > best_doy:
            best_doy = entry_doy
            best_val = hgt

    # Before first entry of year — wrap to last
    if best_val is None:
        best_val = list(_HEIGHT_CONTROL.values())[-1]

    return best_val

_today_kh    = _date_kh.today()
KEY_HGT_500  = _get_key_hgt_500(_today_kh)

print(f'  500 hPa key height line: {KEY_HGT_500} m  (date: {_today_kh})')

# ---- KEY HEIGHT LINES (bold) --------------------------------------------
KEY_HGT_850 = 0000
KEY_HGT_700 = 0000
# KEY_HGT_500 set above ↑
KEY_HGT_250 = 0000

center_lat = 53.3097
center_lon = -113.5797

# ══════════════════════════════════════════════════════════
#  CONTOUR / ISOTHERM STYLE CONTROLS  (edit these freely)
# ══════════════════════════════════════════════════════════

################
### Surface  ###
################

# Isobars (SLP)
CTR_SLP_COLOR       = '#000000'   # line colour
CTR_SLP_WEIGHT      = 1.5         # normal line thickness (px)
CTR_SLP_BOLD_WEIGHT = 2.8         # bold (every 4th contour) thickness
CTR_SLP_OPACITY     = 1.0
CTR_SLP_LABEL_SIZE  = '14px'      # label font size

# Isotherms (T or T-Td)
CTR_ISO_COLOR       = '#b02020'   # line colour
CTR_ISO_WEIGHT      = 0.8
CTR_ISO_BOLD_WEIGHT = 1.2
CTR_ISO_OPACITY     = 1.0
CTR_ISO_LABEL_SIZE  = '12px'       # smaller than isobars

# T-Td (surface moisture) contours
CTR_TTD_COLOR       = '#00aa44'
CTR_TTD_WEIGHT      = 1.0
CTR_TTD_BOLD_WEIGHT = 1.4
CTR_TTD_OPACITY     = 0.7
CTR_TTD_LABEL_SIZE  = '10px'

# Surface T-Td moist fill
CTR_SFC_TTD_MOIST_COLOR      = '#00aa44'
CTR_SFC_TTD_MOIST_WEIGHT     = 2.0
CTR_SFC_TTD_MOIST_OPACITY    = 0.9
CTR_SFC_TTD_MOIST_FILL       = '#90ee90'
CTR_SFC_TTD_MOIST_FILL_OPACITY = 0.3
CTR_SFC_TTD_DRY_WEIGHT       = 1.2
CTR_SFC_TTD_DRY_OPACITY      = 0.7

#################
### Upper Air ###
#################

# UA height contours
CTR_UA_COLOR        = '#000000'
CTR_UA_WEIGHT       = 2.5
CTR_UA_BOLD_WEIGHT  = 5.5
CTR_UA_OPACITY      = 1.0
CTR_UA_LABEL_SIZE   = '14px'

# UA temperature isotherms
CTR_UA_TEMP_WEIGHT      = 0.8
CTR_UA_TEMP_BOLD_WEIGHT = 1.2
CTR_UA_TEMP_OPACITY     = 1.0
CTR_UA_TEMP_LABEL_SIZE  = '12px'

# UA T-Td (dewpoint depression) contours
CTR_UA_TTD_WEIGHT       = 1.1
CTR_UA_TTD_OPACITY      = 0.65
CTR_UA_TTD_LABEL_SIZE   = '10px'

# UA wind speed contours
CTR_UA_SPED_WEIGHT_MIN  = 0.8   # weight at low speeds
CTR_UA_SPED_WEIGHT_MAX  = 2.0   # weight at high speeds
CTR_UA_SPED_OPACITY_MIN = 0.45
CTR_UA_SPED_OPACITY_MAX = 0.90

# 850 hPa moisture fill
CTR_850_MOIST_BORDER_COLOR   = '#cc0000'
CTR_850_MOIST_WEIGHT         = 1.5
CTR_850_MOIST_OPACITY        = 0.9
CTR_850_MOIST_FILL           = '#add8e6'
CTR_850_MOIST_FILL_OPACITY   = 1.0

# Instability (TCU / CB) fills
CTR_INSTAB_TCU_FILL          = '#ffbb77'
CTR_INSTAB_TCU_BORDER        = '#cc6600'
CTR_INSTAB_TCU_FILL_OPACITY  = 0.35
CTR_INSTAB_CB_FILL           = '#ff4400'
CTR_INSTAB_CB_BORDER         = '#cc2200'
CTR_INSTAB_CB_FILL_OPACITY   = 0.55
CTR_INSTAB_WEIGHT            = 1.2
CTR_INSTAB_OPACITY           = 0.85





style_js = (
    '<script>\n'
    'window._SYN_STYLE = {\n'
   f'  slp:  {{ color:"{CTR_SLP_COLOR}",  weight:{CTR_SLP_WEIGHT},  boldWeight:{CTR_SLP_BOLD_WEIGHT},  opacity:{CTR_SLP_OPACITY},  labelSize:"{CTR_SLP_LABEL_SIZE}"  }},\n'
   f'  iso:  {{ color:"{CTR_ISO_COLOR}",  weight:{CTR_ISO_WEIGHT},  boldWeight:{CTR_ISO_BOLD_WEIGHT},  opacity:{CTR_ISO_OPACITY},  labelSize:"{CTR_ISO_LABEL_SIZE}"  }},\n'
   f'  ttd:  {{ color:"{CTR_TTD_COLOR}",  weight:{CTR_TTD_WEIGHT},  boldWeight:{CTR_TTD_BOLD_WEIGHT},  opacity:{CTR_TTD_OPACITY},  labelSize:"{CTR_TTD_LABEL_SIZE}"  }},\n'
   f'  ua:   {{ color:"{CTR_UA_COLOR}",   weight:{CTR_UA_WEIGHT},   boldWeight:{CTR_UA_BOLD_WEIGHT},   opacity:{CTR_UA_OPACITY},   labelSize:"{CTR_UA_LABEL_SIZE}"   }},\n'
   f'  uatemp: {{ weight:{CTR_UA_TEMP_WEIGHT}, boldWeight:{CTR_UA_TEMP_BOLD_WEIGHT}, opacity:{CTR_UA_TEMP_OPACITY}, labelSize:"{CTR_UA_TEMP_LABEL_SIZE}" }},\n'
   f'  uatempbands: {{ base850:{UA_TEMP_BAND_BASE_850}, base500:{UA_TEMP_BAND_BASE_500}, opacity:{UA_TEMP_BAND_OPACITY}, show:{"true" if UA_TEMP_BAND_SHOW else "false"}, bands850:{_json.dumps(UA_TEMP_BANDS_850)}, bands500:{_json.dumps(UA_TEMP_BANDS_500)} }},\n'
   f'  uattd:  {{ weight:{CTR_UA_TTD_WEIGHT}, opacity:{CTR_UA_TTD_OPACITY}, labelSize:"{CTR_UA_TTD_LABEL_SIZE}" }},\n'
   f'  uasped: {{ weightMin:{CTR_UA_SPED_WEIGHT_MIN}, weightMax:{CTR_UA_SPED_WEIGHT_MAX}, opacityMin:{CTR_UA_SPED_OPACITY_MIN}, opacityMax:{CTR_UA_SPED_OPACITY_MAX} }},\n'
   f'  sfcttd: {{ moistColor:"{CTR_SFC_TTD_MOIST_COLOR}", moistWeight:{CTR_SFC_TTD_MOIST_WEIGHT}, moistOpacity:{CTR_SFC_TTD_MOIST_OPACITY}, moistFill:"{CTR_SFC_TTD_MOIST_FILL}", moistFillOpacity:{CTR_SFC_TTD_MOIST_FILL_OPACITY}, dryWeight:{CTR_SFC_TTD_DRY_WEIGHT}, dryOpacity:{CTR_SFC_TTD_DRY_OPACITY} }},\n'
   f'  m850:   {{ borderColor:"{CTR_850_MOIST_BORDER_COLOR}", weight:{CTR_850_MOIST_WEIGHT}, opacity:{CTR_850_MOIST_OPACITY}, fill:"{CTR_850_MOIST_FILL}", fillOpacity:{CTR_850_MOIST_FILL_OPACITY} }},\n'
   f'  instab: {{ tcuFill:"{CTR_INSTAB_TCU_FILL}", tcuBorder:"{CTR_INSTAB_TCU_BORDER}", tcuFillOpacity:{CTR_INSTAB_TCU_FILL_OPACITY}, cbFill:"{CTR_INSTAB_CB_FILL}", cbBorder:"{CTR_INSTAB_CB_BORDER}", cbFillOpacity:{CTR_INSTAB_CB_FILL_OPACITY}, weight:{CTR_INSTAB_WEIGHT}, opacity:{CTR_INSTAB_OPACITY} }}\n'
    '};\n'
    '</script>\n'
)




tooltip_toggle_html = (
    '<style>\n'
    '#syn-tt-btn {\n'
    '  position:fixed;top:10px;left:230px;z-index:10002;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;\n'
    '  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;\n'
    '}\n'
    '#syn-tt-btn.tt-off { color:#999; background:rgba(235,235,235,0.96); }\n'
    '</style>\n'
    '<button id="syn-tt-btn" onclick="synToggleTooltips()">'
    + ('&#x1F4AC; Tooltips: ON' if _TOOLTIP_ON else '&#x1F4AC; Tooltips: OFF')
    + '</button>\n'
    '<script>\n'
    'var _synTooltipsEnabled = ' + ('true' if _TOOLTIP_ON else 'false') + ';\n'
    '\n'
    'L.Layer.include({\n'
    '  _synOrigBindTooltip: L.Layer.prototype.bindTooltip\n'
    '});\n'
    'L.Layer.include({\n'
    '  bindTooltip: function(content, options) {\n'
    '    this._ttContent  = (typeof content === "string") ? content : null;\n'
    '    this._ttOptions  = options || null;\n'
    '    if (!_synTooltipsEnabled) return this;\n'
    '    return this._synOrigBindTooltip(content, options);\n'
    '  }\n'
    '});\n'
    '\n'
    'function synToggleTooltips() {\n'
    '  _synTooltipsEnabled = !_synTooltipsEnabled;\n'
    '  var btn = document.getElementById("syn-tt-btn");\n'
    '  if (_synTooltipsEnabled) {\n'
    '    btn.innerHTML = "&#x1F4AC; Tooltips: ON";\n'
    '    btn.classList.remove("tt-off");\n'
    '  } else {\n'
    '    btn.innerHTML = "&#x1F4AC; Tooltips: OFF";\n'
    '    btn.classList.add("tt-off");\n'
    '  }\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  MAP.eachLayer(function(layer) {\n'
    '    if (_synTooltipsEnabled) {\n'
    '      if (layer._ttContent) {\n'
    '        layer._synOrigBindTooltip(layer._ttContent, layer._ttOptions || undefined);\n'
    '      }\n'
    '    } else {\n'
    '      if (layer.getTooltip && layer.getTooltip()) {\n'
    '        var tt = layer.getTooltip();\n'
    '        if (tt && tt._content && !layer._ttContent) {\n'
    '          layer._ttContent = tt._content;\n'
    '        }\n'
    '        layer.unbindTooltip();\n'
    '      }\n'
    '    }\n'
    '  });\n'
    '}\n'
    '</script>\n'
)

import json as _json_guard

if '_ts_ua_json_str' not in globals():
    print("⚠ _ts_ua_json_str missing — upper-air contours will be empty. Run Cell 7.6 first.")
    _ts_ua_json_str = _json_guard.dumps({})

if '_ts_ua_stn_json_str' not in globals():
    print("⚠ _ts_ua_stn_json_str missing — upper-air stations will be empty.")
    _ts_ua_stn_json_str = _json_guard.dumps({})

if '_ts_slp_json_str' not in globals():
    print("⚠ _ts_slp_json_str missing — per-timestamp SLP will be empty. Run Cell 7.5 first.")
    _ts_slp_json_str = _json_guard.dumps({})

if '_ua_date_map' not in globals():
    print("⚠ _ua_date_map missing — UA date labels will be empty.")
    _ua_date_map = {}


m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=None, prefer_canvas=True)
m.get_root().html.add_child(Element(style_js))
m.get_root().html.add_child(Element(
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>\n'
    '<script>\n'
    'function synShowSlpChart(btn) {\n'
    '  var cid  = btn.getAttribute("data-chartid");\n'
    '  var wrap = document.getElementById("slp-chart-wrap-" + cid);\n'
    '  var shown = wrap.style.display !== "none";\n'
    '  wrap.style.display = shown ? "none" : "block";\n'
    '  btn.textContent = shown ? "📈 P-Tendency chart" : "▲ Hide chart";\n'
    '  if (!shown) {\n'
    '    var ctx = document.getElementById(cid);\n'
    '    if (ctx && !ctx._chartInst) {\n'
    '      var labels = JSON.parse(btn.getAttribute("data-labels"));\n'
    '      var values = JSON.parse(btn.getAttribute("data-values"));\n'
    '      var mn = parseFloat(btn.getAttribute("data-min"));\n'
    '      var mx = parseFloat(btn.getAttribute("data-max"));\n'
    '      var col = btn.getAttribute("data-color");\n'
    '      ctx._chartInst = new Chart(ctx, {\n'
    '        type: "line",\n'
    '        data: { labels: labels, datasets: [{\n'
    '          label: "SLP (hPa)", data: values,\n'
    '          borderColor: col, backgroundColor: col + "22",\n'
    '          pointBackgroundColor: col, pointRadius: 3,\n'
    '          borderWidth: 2, tension: 0.3, fill: true\n'
    '        }]},\n'
    '        options: {\n'
    '          responsive: false,\n'
    '          plugins: { legend: {display:false},\n'
    '            tooltip: {callbacks: {label: function(c){ return c.parsed.y.toFixed(1)+" hPa"; }}}},\n'
    '          scales: {\n'
    '            x: {ticks: {font:{size:8}, maxRotation:45, maxTicksLimit:6}},\n'
    '            y: {min:mn, max:mx, ticks:{font:{size:8}},\n'
    '                title:{display:true, text:"hPa", font:{size:8}}}\n'
    '          }\n'
    '        }\n'
    '      });\n'
    '    }\n'
    '  }\n'
    '}\n'
    '</script>\n'
))

folium.TileLayer(tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attr='CartoDB Positron', name='White (CartoDB)', max_zoom=19).add_to(m)
folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', max_zoom=19).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='ESRI World Topo', name='ESRI Topo', max_zoom=19).add_to(m)
folium.TileLayer(tiles='about:blank', attr='Blank', name='Blank (borders only)', max_zoom=19).add_to(m)

m.get_root().html.add_child(Element(
    '<style>.leaflet-container{background:#ffffff!important;}</style>\n'
    '<script>\n'
    '(function(){\n'
    '  function initBlank(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(initBlank,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    var blankLayer=null;\n'
    '    var others=[];\n'
    '    MAP.eachLayer(function(l){\n'
    '      if(l instanceof L.TileLayer){\n'
    '        if(l.options.name==="Blank (borders only)" || (l._url&&l._url==="about:blank")){\n'
    '          blankLayer=l;\n'
    '        } else {\n'
    '          others.push(l);\n'
    '        }\n'
    '      }\n'
    '    });\n'
    '    others.forEach(function(l){MAP.removeLayer(l);});\n'
    '    if(blankLayer) blankLayer.addTo(MAP);\n'
    '  }\n'
    '  function tryInitBlank(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(tryInitBlank,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    MAP.on("layeradd", function(){\n'
    '      MAP.eachLayer(function(l){\n'
    '        if(l instanceof L.TileLayer && l._url && l._url!=="about:blank"){\n'
    '          MAP.removeLayer(l);\n'
    '        }\n'
    '      });\n'
    '    });\n'
    '    initBlank();\n'
    '  }\n'
    '  if(document.readyState==="complete"){setTimeout(tryInitBlank,100);}\n'
    '  else{window.addEventListener("load",function(){setTimeout(tryInitBlank,100);});}\n'
    '})();\n'
    '</script>\n'
))

# ---- SLP CONTOURS --------------------------------------------------------
slp_fg = folium.FeatureGroup(name='SLP Isobars', show=True)
# NEW
if slp_grid is not None:
    _ny, _nx = slp_grid.shape
    _lv  = np.linspace(lon_vec[0], lon_vec[-1], _nx)
    _ltv = np.linspace(lat_vec[0], lat_vec[-1], _ny)
    glon, glat = np.meshgrid(_lv, _ltv)
    slp_min = np.floor(slp_grid.min() / SLP_INTERVAL) * SLP_INTERVAL
    slp_max = np.ceil(slp_grid.max()  / SLP_INTERVAL) * SLP_INTERVAL
    levels  = np.arange(slp_min, slp_max + SLP_INTERVAL, SLP_INTERVAL)
    fig_c, ax_c = plt.subplots(figsize=(1, 1))
    cs = ax_c.contour(glon, glat, slp_grid, levels=levels)
    plt.close(fig_c)
    for lvl_idx, lvl in enumerate(cs.levels):
        for coords in cs.allsegs[lvl_idx]:
            if len(coords) < 2: continue
            geo_coords = [[float(c[0]), float(c[1])] for c in coords]
            feature = {'type':'Feature','geometry':{'type':'LineString','coordinates':geo_coords},'properties':{'level':float(lvl)}}
            folium.GeoJson(feature, style_function=lambda f: {'color':'#000000','weight':1,'opacity':0.7},
                tooltip=folium.Tooltip(f'{int(lvl)} ')).add_to(slp_fg)
            if int(lvl) % 4 == 0 and len(coords) > 4:
                mid = coords[len(coords)//2]
                folium.Marker(location=[float(mid[1]), float(mid[0])],
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:9px;font-weight:900;color:#1a3a6a;'
                              f'font-family:Courier New,monospace;white-space:nowrap;'
                              f'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">'
                              f'{int(lvl)}</div>'),
                        icon_size=(32,14), icon_anchor=(16,7))).add_to(slp_fg)

# ----Surface H/L MARKERS ---------------------------------------------------------
hl_fg = folium.FeatureGroup(name='H/L Centers', show=True)
if hl_centers:
    for c in hl_centers:
        color  = 'black'
        shadow = '1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white'
        html = (f'<div style="display:flex;flex-direction:column;align-items:center;pointer-events:none">'
                f'<div style="font-size:28px;font-weight:900;color:{color};'
                f'font-family:Arial Black,sans-serif;line-height:1;text-shadow:{shadow};">{c["type"]}</div>'
                f'</div>')
        folium.Marker(location=[c['lat'], c['lon']],
            icon=folium.DivIcon(html=html, icon_size=(60,54), icon_anchor=(30,12)),
            tooltip=f'{c["type"]} {c["val"]:.1f} hPa').add_to(hl_fg)

# ---- BUILD PER-TIMESTAMP STATION DATA ------------------------------------
import json as _json2
_ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
# ── Tendency key audit ──────────────────────────────────────────────────
_tend_debug = sum(1 for d in metar_records if d.get('tendency') is not None)
_notend_debug = sum(1 for d in metar_records if d.get('tendency') is None)
print(f'tendency audit: {_tend_debug} with tendency, {_notend_debug} without (None/missing)')
if _tend_debug == 0:
    print('  ⚠ NO tendency data found — check Cell 5b ran after Cell 4')
else:
    from collections import Counter
    print('  sample:', Counter(d.get('tendency') for d in metar_records[:20]))
# ────────────────────────────────────────────────────────────────────────
# ensure keys always exist regardless of cell run order
for _mr in metar_records:
    _mr.setdefault('tendency', None)
    _mr.setdefault('pressure_change', None)

_ts_data = {}
for _ts in _ts_all:
    _entries = []
    for _d in metar_records:
        if _d['timestamp'] != _ts: continue
        _fc = {'VFR':'green','MVFR':'steelblue','IFR':'crimson','LIFR':'red'}.get(_d['flt_cat'], '#888')
        _wg = f' G{_d["wind_gust"]}' if _d.get('wind_gust') else ''
        _tend_raw = _d.get('tendency')
        _pc_raw   = _d.get('pressure_change')
        _TEND_SYM = {
            'rising':'/', 'falling':'\\', 'steady':'—',
            'rising_falling':'∧', 'falling_rising':'V',
            'rising_steady':'⌐', 'falling_steady':'∟',
        }
        _TEND_LABEL = {
            'rising':'Rising', 'falling':'Falling', 'steady':'Steady',
            'rising_falling':'Rising then falling', 'falling_rising':'Falling then rising',
            'rising_steady':'Rising then steady', 'falling_steady':'Falling then steady',
        }
        if _tend_raw:
            _sym    = _TEND_SYM.get(_tend_raw, '?')
            _lbl    = _TEND_LABEL.get(_tend_raw, _tend_raw)
            _pc_str = ''
            if _pc_raw is not None and _tend_raw != 'steady':
                _sign   = '+' if _pc_raw > 0 else ''
                _pc_str = f' ({_sign}{_pc_raw/10:.1f} hPa)'
            _tend_html = (f'<span style="font-weight:bold;font-size:13px;'
                          f'font-family:Courier New,monospace">{_sym}</span>'
                          f' {_lbl}{_pc_str}')
        else:
            _tend_html = '<span style="color:#aaa">insufficient history</span>'
        # ── SLP history sparkline (data- attributes only, no inline script) ──
        _slp_hist = sorted(
            [(r['timestamp'], r['slp']) for r in metar_records
             if r['icao'] == _d['icao'] and r['slp'] is not None
             and r['timestamp'] <= _ts],
            key=lambda x: x[0]
        )
        _chart_id = f'slp{_d["icao"]}{_ts}'.replace(' ','').replace(':','').replace('-','')
        if len(_slp_hist) >= 2:
            _ch_labels = _json2.dumps([x[0] for x in _slp_hist])
            _ch_values = _json2.dumps([x[1] for x in _slp_hist])
            _ch_min    = round(min(x[1] for x in _slp_hist) - 1, 1)
            _ch_max    = round(max(x[1] for x in _slp_hist) + 1, 1)
            _ch_color  = (_tend_color or '#1a4a8a')
            _chart_html = (
                f'<a href="javascript:void(0)" '
                f'data-chartid="{_chart_id}" '
                f'data-labels=\'{_ch_labels}\' '
                f'data-values=\'{_ch_values}\' '
                f'data-min="{_ch_min}" data-max="{_ch_max}" data-color="{_ch_color}" '
                f'style="font-size:10px;color:#1a4a8a;cursor:pointer;text-decoration:underline;" '
                f'onclick="synShowSlpChart(this)">📈 P-Tendency chart</a><br>'
                f'<div id="slp-chart-wrap-{_chart_id}" style="display:none;margin-top:4px;">'
                f'<canvas id="{_chart_id}" width="240" height="110" '
                f'style="width:240px;height:110px;display:block;"></canvas>'
                f'</div>'
            )
        else:
            _chart_html = '<span style="font-size:10px;color:#aaa">📈 P-Tendency chart (insufficient history)</span><br>'

        _pop = (f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
                f'<b style="font-size:14px;color:#1a4a8a">{_d["icao"]}</b> '
                f'<span style="color:{_fc};font-weight:bold">{_d["flt_cat"]}</span><br>'
                f'<span style="color:#888;font-size:10px">{_d["name"]}</span>'
                f'<hr style="margin:4px 0">'
                f'Temp/Dew: <b>{_d["temp"]}C / {_d["dew"]}C</b><br>'
                f'Wind: <b>{_d["wind_dir"]}/{_d["wind_spd"]}kt{_wg}</b><br>'
                f'Vis: <b>{_d["vis"]} SM</b> Wx: <b>{_d["weather"] or "NIL"}</b><br>'
                f'SLP: <b>{_d["slp"]} hPa</b> RH: <b>{_d["rh"]}%</b><br>'
                f'Tendency: {_tend_html}<br>'
                f'Cloud: <b>' + ' '.join(c['raw'] for c in _d['clouds']) + '</b><br>'
                + _chart_html
                + f'<a href="https://aviationweather.gov/api/data/metar?ids={_d["icao"]}&hours=24&taf=1" '
                f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {_d["icao"]} ↗</a></div>')




        _svg_str, _sw, _sh = station_model_svg({**_d, 'is_surface': True}, S=34)
        _ttd_val = round(_d['temp'] - _d['dew'], 1) if _d.get('temp') is not None and _d.get('dew') is not None else None
        _pc_hpa  = (_d.get('pressure_change') or 0) / 10.0
        if   _pc_hpa <= -3:              _tend_color = '#8B0000'  # dark red
        elif _pc_hpa <= -2:              _tend_color = '#cc0000'  # red
        elif _pc_hpa <= -1:              _tend_color = '#ff6666'  # light red
        elif _pc_hpa >=  3:              _tend_color = '#00008B'  # dark blue
        elif _pc_hpa >=  2:              _tend_color = '#1a4a8a'  # blue
        elif _pc_hpa >=  1:              _tend_color = '#66aaff'  # light blue
        else:                            _tend_color = None
        _entries.append({'lat':_d['lat'],'lon':_d['lon'],'popup':_pop,
            'tip':f'{_d["icao"]} {_d["temp"]}C/{_d["dew"]}C {_d["wind_dir"]}/{_d["wind_spd"]}kt',
            'svg': _svg_str, 'svg_w': int(_sw), 'svg_h': int(_sh),
            'ttd': _ttd_val, 'tend_color': _tend_color})
    _ts_data[_ts] = _entries
_ts_json_str = _json2.dumps(_ts_data)
_ts_list_str = _json2.dumps(_ts_all)
_latest_ts   = _ts_all[-1] if _ts_all else ''

folium.LayerControl(collapsed=False).add_to(m)

# ---- BUILD UA STATION DATA -----------------------------------------------
import json as _json3
_ua_stn_data = {}
for _hour, _grp in ua_summary_df.groupby('hour'):
    _key = str(int(_hour))
    _stns = []
    for _, _r in _grp.iterrows():
        def _fmt(v, dec=1):
            return f'{v:.{dec}f}' if v is not None and not (isinstance(v, float) and __import__("math").isnan(v)) else '—'
        def _fmti(v):
            return f'{int(round(v))}' if v is not None and not (isinstance(v, float) and __import__("math").isnan(v)) else '—'
        _pop = (f'<div style="font-family:monospace;font-size:11px;min-width:240px">'
        f'<b style="font-size:13px;color:#cc6600">{_r["icao"]}</b> '
        f'<span style="color:#888;font-size:10px">{_r["stn_name"]}</span><br>'
        f'<span style="color:#888;font-size:10px">'
        f'Lat: <b>{float(_r["lat"]):.2f}°N</b> &nbsp; '
        f'Lon: <b>{float(_r["lon"]):.2f}°E</b> &nbsp; '
        f'WMO: <b>{_r["wmo"]}</b>'
        f'</span>'
        f'<hr style="margin:4px 0">')

        for _lvl in [850, 700, 500, 250]:
            _h   = _fmti(_r.get(f'HGHT_{_lvl}'))
            _t   = _fmt(_r.get(f'TEMP_{_lvl}'))
            _td  = _fmt(_r.get(f'DWPT_{_lvl}'))
            _tv, _tdv = _r.get(f'TEMP_{_lvl}'), _r.get(f'DWPT_{_lvl}')
            _ttd = (f'{_tv - _tdv:.1f}' if _tv is not None and _tdv is not None
                    and not (isinstance(_tv, float) and __import__("math").isnan(_tv))
                    and not (isinstance(_tdv, float) and __import__("math").isnan(_tdv)) else '—')
            _wd  = _fmti(_r.get(f'DRCT_{_lvl}'))
            _ws  = _fmt(_r.get(f'SPED_{_lvl}'))
            _pop += (f'<b style="color:#cc6600">{_lvl} hPa</b> '
                     f'Hgt:<b>{_h}m</b> T:<b>{_t}°C</b> '
                     f'Td:<b>{_td}°C</b> T-Td:<b>{_ttd}°C</b> '
                     f'Wnd:<b>{_wd}/{_ws}m/s</b><br>')
        _vt = str(_r["valid_time"])[:10]
        _hr = _key
        _sounding_url = (f'https://weather.uwyo.edu/wsgi/sounding'
                         f'?datetime={_vt}%20{_hr}:00:00&id={_r["wmo"]}&src=BUFR&type=PNG:SKEWT')
        _t5 = _r.get('TEMP_500')
        _t7 = _r.get('TEMP_700')
        _instab_str = '—'
        _instab_cat = ''
        if (_t5 is not None and _t7 is not None
                and not (isinstance(_t5, float) and __import__("math").isnan(_t5))
                and not (isinstance(_t7, float) and __import__("math").isnan(_t7))):
            _tdiff = _t7 - _t5
            _instab_str = f'{_tdiff:.1f}'
            if _tdiff >= 18:
                _instab_cat = ' <span style="color:#cc2200;font-weight:bold">CB</span>'
            elif _tdiff >= 16:
                _instab_cat = ' <span style="color:#cc5500;font-weight:bold">TCU</span>'
        _pop += (f'<hr style="margin:4px 0">T700-500: <b>{_instab_str}°C</b>{_instab_cat}<br>')
        _pop += (f'<a href="{_sounding_url}" target="_blank" style="font-size:10px;color:#cc6600;">Sounding ↗</a></div>')
        _level_svgs = {}
        for _lvl in [850, 700, 500, 250]:
            _lt  = _r.get(f'TEMP_{_lvl}')
            _ltd = _r.get(f'DWPT_{_lvl}')
            _lwd = _r.get(f'DRCT_{_lvl}')
            _lws = _r.get(f'SPED_{_lvl}')
            _lh  = _r.get(f'HGHT_{_lvl}')
            _lttd = None
            if _lt is not None and _ltd is not None and not (isinstance(_lt, float) and __import__("math").isnan(_lt)) and not (isinstance(_ltd, float) and __import__("math").isnan(_ltd)):
                _lttd = round(_lt - _ltd, 1)
            _lh_label = ''
            if _lh is not None and not (isinstance(_lh, float) and __import__("math").isnan(_lh)):
                _dam_f = _lh / 10.0
                _dam_str = f'{_dam_f:.1f}'
                _dot = _dam_str.index('.')
                _lh_label = str(int(round(_lh / 10)))[1:]
            _lws_kt = None
            if _lws is not None and not (isinstance(_lws, float) and __import__("math").isnan(_lws)):
                _lws_kt = _lws * 1.94384
            _ua_d = {
                'icao': str(_r['icao']),
                'temp': round(_lt, 1) if _lt is not None and not (isinstance(_lt, float) and __import__("math").isnan(_lt)) else None,
                'dew':  round(_lttd, 1) if _lttd is not None else None,
                'wind_dir': int(_lwd) if _lwd is not None and not (isinstance(_lwd, float) and __import__("math").isnan(_lwd)) else None,
                'wind_spd': _lws_kt, 'wind_gust': 0,
                'vis': None, 'weather': '', 'slp_label': _lh_label,
                'oktas': 8, 'has_sky_obs': True, 'clouds': [], 'lowest_sig': None,
                'ceiling': 99999, 'flt_cat': 'VFR',
                'lat': 0, 'lon': 0, 'timestamp': '', 'rh': 0,
                'tendency': None, 'pressure_change': None,
                'is_surface': False,   # ← add this
            }
            _svg_str, _sw, _sh = station_model_svg(_ua_d, S=34)
            _level_svgs[str(_lvl)] = {'svg': _svg_str, 'w': int(_sw), 'h': int(_sh)}
        _stns.append({
            'lat': float(_r['lat']), 'lon': float(_r['lon']),
            'icao': str(_r['icao']), 'name': str(_r['stn_name']),
            'popup': _pop,
            'tip': f'{_r["icao"]} {_r["stn_name"]} | 850:{_fmt(_r.get("TEMP_850"))}°C 500:{_fmt(_r.get("TEMP_500"))}°C'
                + (f' | T700-500:{_fmt(_r.get("TEMP_700") - _r.get("TEMP_500"),1)}°C'
                   if _r.get("TEMP_700") is not None and _r.get("TEMP_500") is not None
                   and not (isinstance(_r.get("TEMP_700"), float) and __import__("math").isnan(_r.get("TEMP_700")))
                   and not (isinstance(_r.get("TEMP_500"), float) and __import__("math").isnan(_r.get("TEMP_500")))
                   else ''),
            'svgs': _level_svgs,
        })
    _ua_stn_data[_key] = _stns
_ts_ua_stn_json_str = _json3.dumps(_ua_stn_data)

# ---- BUTTON BAR ----------------------------------------------------------

ts_bar_html = (
    '<div id="syn-ts-bar" style="'
    'position:fixed;bottom:0;left:0;right:0;z-index:10000;'
    'background:rgba(255,255,255,0.96);border-top:1px solid #ccc;'
    'padding:4px 8px;font-family:Courier New,monospace;font-size:12px;'
    'box-shadow:0 -2px 10px rgba(0,0,0,0.12);display:grid;'
    'grid-template-columns:auto auto 1fr;gap:3px 8px;align-items:center;'
    'overflow-x:auto;overflow-y:hidden;max-height:none;">'

    # ── COL 1 TOP: UA time filter ──
    '<div style="display:flex;align-items:center;gap:4px;white-space:nowrap;">'
    '<b style="color:#555;font-size:9px">UA</b>'
    '<button id="btn-f00" onclick="synFilterHour(0)" '
'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#4a7fc1;color:#fff">-- 00Z</button>'
    '<button id="btn-f12" onclick="synFilterHour(12)" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#4a7fc1;color:#fff">12Z</button>'
    '</div>'

    # ── COL 2 TOP: UA controls ──
    '<div style="display:flex;align-items:center;gap:4px;white-space:nowrap;">'
    '<b style="color:#555;font-size:9px">UA</b>'
    '<select id="ua-level-sel" onchange="synUpdateUALevel(this.value)" '
    'style="font-family:Courier New,monospace;font-size:10px;padding:1px 4px;'
    'border:1px solid #aac;border-radius:3px;background:#f8fbff;color:#1a4a8a;">'
    '<option value="">-- level --</option>'
    '<option value="850">850</option>'
    '<option value="700">700</option>'
    '<option value="500">500</option>'
    '<option value="250">250</option>'
    '</select>'
    '<button id="btn-ua-stns" onclick="synToggleUAStns()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#4a7fc1;color:#fff">Stns</button>'
    '<button id="btn-ua-hght" onclick="synToggleUA(\'hght\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Hght</button>'
    '<button id="btn-ua-temp" onclick="synToggleUA(\'temp\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#4a7fc1;color:#fff">Temp</button>'
    '<button id="btn-ua-ttdp" onclick="synToggleUA(\'ttdp\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">T-Td</button>'
    '<button id="btn-ua-sped" onclick="synToggleUA(\'sped\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Wind</button>'
    '<button id="btn-ua-tbands" onclick="synToggleUATempBands()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Fill</button>'
    '</div>'

    # ── COL 3 TOP: Ridge / Trough ──
      '<div style="display:flex;align-items:center;gap:4px;flex-wrap:nowrap;white-space:nowrap;">'
    '<b style="color:#555;font-size:9px">ANALYSIS</b>'
    '<button id="btn-ttd" onclick="synToggleLayer(\'ttd\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Sfc Moist</button>'
    '<button id="btn-850moist" onclick="synToggle850Moist()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">850 Moist</button>'
    '<button id="btn-700moist" onclick="synToggle700Moist()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">700 Moist</button>'
    '<button id="btn-500moist" onclick="synToggle500Moist()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">500 Moist</button>'
    '<button id="btn-850dry" onclick="synToggle850Dry()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">850 Dry</button>'
    '<button id="btn-700dry" onclick="synToggle700Dry()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">700 Dry</button>'
    '<button id="btn-500dry" onclick="synToggle500Dry()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">500 Dry</button>'
    '<button id="btn-vort" onclick="synToggleVort()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Vort500</button>'
    '<button id="btn-tend-ring" onclick="synToggleTendRing()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">P-Tend</button>'


    '</div>'

    # ── COL 1 BOTTOM: METAR time dropdown ──
    '<div style="display:flex;align-items:center;gap:4px;white-space:nowrap;">'
    '<b style="color:#555;font-size:9px">METAR</b>'
    '<select id="ts-select" onchange="synUpdateTS(this.value)" '
    'style="font-family:Courier New,monospace;font-size:11px;padding:2px 4px;'
    'border:1px solid #aac;border-radius:4px;background:#f8fbff;color:#1a4a8a;cursor:pointer;max-width:90px"></select>'
    '<span id="ts-count" style="color:#888;font-size:10px;min-width:40px"></span>'
    '</div>'

    # ── COL 2 BOTTOM: SFC controls ──
    '<div style="display:flex;align-items:center;gap:4px;white-space:nowrap;">'
    '<b style="color:#555;font-size:9px">SFC</b>'
    '<select id="sfc-master-sel" onchange="synSfcMaster(this.value)" '
    'style="font-family:Courier New,monospace;font-size:10px;padding:1px 4px;'
    'border:1px solid #aac;border-radius:3px;background:#f8fbff;color:#1a4a8a;">'
    '<option value="on">On</option>'
    '<option value="off" selected>Off</option>'
    '</select>'
    '<button id="btn-sfcmod" onclick="synToggleSfcModel()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Sfc Stn</button>'
    '<button id="btn-ua-avail" onclick="synToggleUAAvail()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #1a3a6a;border-radius:3px;background:#1a3a6a;color:#fff;font-weight:bold;">&#128225; UA Sonde Avail</button>'
    '<button id="btn-slp" onclick="synToggleLayer(\'slp\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Isobars+H/L</button>'
    '<button id="btn-tmp" onclick="synToggleLayer(\'tmp\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Temp</button>'
    '<button id="btn-sfc-ttd" onclick="synToggleSfcTtd()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">T-Td</button>'
    '<button id="btn-hl"  style="display:none"></button>'
    '<button id="btn-fire-zones" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Fire Zones</button>'
    '</div>'



    # ── COL 3 BOTTOM: Sfc Moist / 850 Moist / Instab ──
    '<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;">'
    '<b style="color:#555;font-size:9px;margin-right:40px">-</b>'
    '<button id="btn-sfc-trough" onclick="synToggleSfcTrough()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">SFC Trgh</button>'
    '<button id="btn-trough" onclick="synToggleThermalTrough()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">850 Trough</button>'
    '<button id="btn-trough700" onclick="synToggleTrough700()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">700 Trough</button>'
    '<button id="btn-trough500" onclick="synToggleTrough500()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">500 Trough</button>'
    '<button id="btn-thermal" onclick="synToggleThermalRidge()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">850 Ridge</button>'
    '<button id="btn-ridge700" onclick="synToggleRidge700()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">700 Ridge</button>'
    '<button id="btn-ridge500" onclick="synToggleRidge500()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">500 Ridge</button>'
    '<button id="btn-instab" onclick="synToggleInstab()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">Instab</button>'
    '<button id="btn-conv-sfc" onclick="synToggleConvSfc()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">SFC Conv</button>'
    '<button id="btn-conv-850" onclick="synToggleConv850()" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;border-radius:3px;background:#b0b8c8;color:#fff">850 Conv</button>'


    '</div>'

    '</div>'
)

m.get_root().html.add_child(Element(ts_bar_html))

# ---- TIMESTEP JS ---------------------------------------------------------
# Helper: _synBtnOn(id)  → blue   _synBtnOff(id) → grey
# No button ever changes its text label.
ts_js = (
    '<script>\n'
    'var KEY_HGT_DAM = {"850":' + str(int(KEY_HGT_850/10)) + ',"700":' + str(int(KEY_HGT_700/10)) + ',"500":' + str(int(KEY_HGT_500/10)) + ',"250":' + str(int(KEY_HGT_250/10)) + '};\n'
    'var KEY_HGT_M   = {"850":' + str(int(KEY_HGT_850)) + ',"700":' + str(int(KEY_HGT_700)) + ',"500":' + str(int(KEY_HGT_500)) + ',"250":' + str(int(KEY_HGT_250)) + '};\n'
    'var _SYN_TS_DATA = ' + _ts_json_str + ';\n'
    'var _SYN_TS_LIST = ' + _ts_list_str + ';\n'
    'var _SYN_SLP     = ' + _ts_slp_json_str + ';\n'
    'var _SYN_UA      = ' + _ts_ua_json_str + ';\n'
    'var _SYN_UA_DATES = ' + _json_guard.dumps(_ua_date_map) + ';\n'

    # ── shared button helpers ──
    'function _synBtnOn(id)  { var b=document.getElementById(id); if(b){b.style.background="#4a7fc1";b.style.color="#fff";} }\n'
    'function _synBtnOff(id) { var b=document.getElementById(id); if(b){b.style.background="#b0b8c8";b.style.color="#fff";} }\n'

    'var _synStnLayer = null;\n'
    'var _synSlpLayer = null;\n'
    'var _synHLLayer  = null;\n'
    'var _synTtdLayer = null;\n'
    'var _synTmpLayer = null;\n'

    'function synUpdateTS(ts) {\n'
    '  var entries = _SYN_TS_DATA[ts] || [];\n'
    '  var countEl = document.getElementById("ts-count");\n'
    '  if (countEl) countEl.textContent = entries.length + " stns";\n'
    '  var dispEl = document.getElementById("syn-ts-display");\n'
    '  if (dispEl) {\n'
    '    var _dm = ts.match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
'    var _months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];\n'
'    var _now=new Date();\n'
'    var _metar_label = _dm ? (_months[_now.getMonth()]+" "+parseInt(_dm[1],10)+" "+_dm[2]+"Z") : ts;\n'
'    var _uaKey = _synUAHourKey==="12" ? "12" : "0";\n'
'    var _uaDt = (_SYN_UA_DATES||{})[_uaKey];\n'
'    var _ua_label;\n'
'    if (_uaDt) {\n'
'      var _dp=_uaDt.match(/(\\d{4})-(\\d{2})-(\\d{2})/);\n'
'      _ua_label = _dp ? _months[parseInt(_dp[2],10)-1]+" "+parseInt(_dp[3],10)+" "+(_synUAHourKey==="12"?"12":"00")+"Z" : (_synUAHourKey==="12"?"12Z":"00Z");\n'
'    } else {\n'
'      _ua_label = _synUAHourKey==="12" ? "12Z" : "00Z";\n'
'    }\n'
'    dispEl.textContent = "METAR: " + _metar_label + "  |  UA:" + _ua_label;\n'
    '  }\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { console.warn("synUpdateTS: map not ready"); return; }\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!MAP || typeof MAP.removeLayer !== "function") return;\n'
    '  if (_synStnLayer)      { MAP.removeLayer(_synStnLayer);      _synStnLayer      = null; }\n'
    '  if (_synSlpLayer)      { MAP.removeLayer(_synSlpLayer);      _synSlpLayer      = null; }\n'
    '  if (_synHLLayer)       { MAP.removeLayer(_synHLLayer);       _synHLLayer       = null; }\n'
    '  if (_synTtdLayer)      { MAP.removeLayer(_synTtdLayer);      _synTtdLayer      = null; }\n'
    '  if (_synTmpLayer)      { MAP.removeLayer(_synTmpLayer);      _synTmpLayer      = null; }\n'
    '  if (_synSfcModelLayer) { MAP.removeLayer(_synSfcModelLayer); _synSfcModelLayer = null; }\n'

    '  _synStnLayer = L.layerGroup();\n'
    '  entries.forEach(function(d) {\n'
    '    L.marker([d.lat, d.lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: \'<div style="width:8px;height:8px;background:#1a4a8a;border-radius:50%;border:1px solid #fff;"></div>\',\n'
    '        iconSize:[8,8], iconAnchor:[4,4], className:""\n'
    '      }), zIndexOffset:100\n'
    '    }).bindPopup(d.popup,{maxWidth:280,closeButton:true}).bindTooltip(d.tip).addTo(_synStnLayer);\n'
    '  });\n'
    '  if (_synShowStations) _synStnLayer.addTo(MAP);\n'


    '  _synSfcModelLayer = L.layerGroup();\n'
    '  entries.forEach(function(d) {\n'
    '    if (!d.svg) return;\n'
    '    L.marker([d.lat, d.lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: d.svg,\n'
    '        iconSize: [d.svg_w || 60, d.svg_h || 60],\n'
    '        iconAnchor: [Math.round((d.svg_w||60)/2), Math.round((d.svg_h||60)/2)],\n'
    '        className: ""\n'
    '      }), zIndexOffset: 150\n'
    '    }).bindPopup(d.popup,{maxWidth:280,closeButton:true}).bindTooltip(d.tip).addTo(_synSfcModelLayer);\n'
    '  });\n'
    '  if (_synShowSfcModel) _synSfcModelLayer.addTo(MAP);\n'
    '  if (_synTendRingLayer) { MAP.removeLayer(_synTendRingLayer); _synTendRingLayer = null; }\n'
    '  _synTendRingLayer = L.layerGroup();\n'
    '  entries.forEach(function(d) {\n'
    '    if (!d.tend_color) return;\n'
    '    L.circleMarker([d.lat, d.lon], {\n'
    '      radius: 12, color: d.tend_color, weight: 2.5,\n'
    '      fillColor: d.tend_color, fillOpacity: 0.25, opacity: 0.95\n'
    '    }).addTo(_synTendRingLayer);\n'
    '  });\n'
    '  if (_synShowTendRing) _synTendRingLayer.addTo(MAP);\n'
    '  if (_synMoistRingLayer) { MAP.removeLayer(_synMoistRingLayer); _synMoistRingLayer = null; }\n'
    '  _synMoistRingLayer = L.layerGroup();\n'
    '  entries.forEach(function(d) {\n'
    '    if (d.ttd === null || d.ttd === undefined || d.ttd > 2) return;\n'
    '    L.circleMarker([d.lat, d.lon], {\n'
    '      radius: 10, color: "#22cc44", weight: 1.5,\n'
    '      fillColor: "#22cc44", fillOpacity: 0.22, opacity: 0.85\n'
    '    }).addTo(_synMoistRingLayer);\n'
    '  });\n'
    '  if (_synShowTtd) _synMoistRingLayer.addTo(MAP);\n'
    '  var slpData = _SYN_SLP[ts] || {contours:[], hl:[]};\n'
    '  _synSlpLayer = L.layerGroup();\n'
    '  slpData.contours.forEach(function(ct) {\n'
    '    var latlngs = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    var _slpS = window._SYN_STYLE.slp;\n'
    '    var _slpBold = (Math.round(ct.level) % 4 === 0);\n'
    '    L.polyline(latlngs, {color:_slpS.color, weight:_slpBold?_slpS.boldWeight:_slpS.weight, opacity:_slpS.opacity})\n'
    '     .bindTooltip(Math.round(ct.level)+" ").addTo(_synSlpLayer);\n'
    '    if (_slpBold) {\n'
    '      L.marker([ct.label_lat, ct.label_lon], {\n'
    '        icon: L.divIcon({\n'
    '          html: \'<div style="font-size:\'+_slpS.labelSize+\';font-weight:bold;color:\'+_slpS.color+\';\'\n'
    '               +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '               +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '               + Math.round(ct.level) + \'</div>\',\n'
    '          iconSize:[42,18], iconAnchor:[21,9], className:""\n'
    '        })\n'
    '      }).addTo(_synSlpLayer);\n'
    '    }\n'
    '  });\n'
    '  if (_synShowSlp) _synSlpLayer.addTo(MAP);\n'

    '  _synHLLayer = L.layerGroup();\n'
    '  slpData.hl.forEach(function(c) {\n'
    '    var color = "black";\n'
    '    var shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";\n'
    '    var html = \'<div style="display:flex;flex-direction:column;align-items:center;">\'\n'
'             + \'<div style="font-size:56px;font-weight:900;color:\'+color+\';\'\n'
'             + \'font-family:Arial Black,sans-serif;line-height:1;text-shadow:\'+shadow+\';">\'+c.type+\'</div>\'\n'
'             + \'</div>\';\n'
'    L.marker([c.lat, c.lon], {\n'
'      icon: L.divIcon({html:html, iconSize:[100,90], iconAnchor:[50,22], className:""}),\n'

    '      zIndexOffset: 200\n'
    '    }).bindTooltip(c.type+" "+c.val+" ").addTo(_synHLLayer);\n'
    '  });\n'
    '  if (_synShowHL) _synHLLayer.addTo(MAP);\n'

    '  _synTtdLayer = L.layerGroup();\n'
    '  (slpData.ttd_contours || []).forEach(function(ct) {\n'
    '    var latlngs = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    var isMoist = ct.level <= 2;\n'
    '    var _sfcTtdS = window._SYN_STYLE ? window._SYN_STYLE.sfcttd : {moistColor:"#00aa44",moistWeight:2,moistOpacity:0.9,moistFill:"#90ee90",moistFillOpacity:0.3,dryWeight:1.2,dryOpacity:0.7};\n'
    '    if (isMoist) {\n'
    '      L.polygon(latlngs, {color:_sfcTtdS.moistColor, weight:_sfcTtdS.moistWeight, opacity:_sfcTtdS.moistOpacity, fillColor:_sfcTtdS.moistFill, fillOpacity:_sfcTtdS.moistFillOpacity})\n'
'       .bindTooltip("T-Td="+ct.level.toFixed(1)+"°C (moist)").addTo(_synTtdLayer);\n'
'    } else {\n'
'      L.polyline(latlngs, {color:_sfcTtdS.moistColor, weight:_sfcTtdS.dryWeight, opacity:_sfcTtdS.dryOpacity, dashArray:"4 4"})\n'
    '       .bindTooltip("T-Td="+ct.level.toFixed(1)+"°C").addTo(_synTtdLayer);\n'
    '    }\n'
    '    if (Math.round(ct.level) % 4 === 0) {\n'
    '      var ttdCol = isMoist ? "#006622" : "#00aa44";\n'
    '      L.marker([ct.label_lat, ct.label_lon], {\n'
    '        icon: L.divIcon({\n'
    '          html: \'<div style="font-size:\'+window._SYN_STYLE.ttd.labelSize+\';font-weight:bold;color:\'+ttdCol+\';\'\n'
    '               +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '               +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '               + Math.round(ct.level) + \'</div>\',\n'
    '          iconSize:[28,16], iconAnchor:[14,8], className:""\n'
    '        })\n'
    '      }).addTo(_synTtdLayer);\n'
    '    }\n'
    '  });\n'
    '  if (_synShowTtd) _synTtdLayer.addTo(MAP);\n'
    '  _synSfcTtdLayer = _synTtdLayer;\n'
    '  if (_synShowSfcTtd) _synSfcTtdLayer.addTo(MAP);\n'

    '  _synTmpLayer = L.layerGroup();\n'
    '  (slpData.tmp_contours || []).forEach(function(ct) {\n'
    '    var latlngs = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    var t = ct.level;\n'
    '    var _isoS = window._SYN_STYLE.iso;\n'
    '    var lineCol;\n'
    '    if (t > 0) { var frac=Math.min(t/40,1); lineCol="rgb("+Math.round(180+75*frac)+",0,0)"; }\n'
    '    else if (t < 0) { var frac=Math.min(Math.abs(t)/40,1); lineCol="rgb(0,0,"+Math.round(180+75*frac)+")"; }\n'
    '    else { lineCol="#00bb00"; }\n'
    '    var isBold = (Math.round(t) % 10 === 0);\n'
    '    L.polyline(latlngs, {color:lineCol, weight:isBold?_isoS.boldWeight:_isoS.weight, opacity:isBold?_isoS.opacity:_isoS.opacity*0.8, dashArray:isBold?null:"6 3"})\n'
    '     .bindTooltip("Temp="+Math.round(t)+"°C").addTo(_synTmpLayer);\n'
    '    L.marker([ct.label_lat, ct.label_lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: \'<div style="font-size:\'+_isoS.labelSize+\';font-weight:\'+( isBold?"900":"bold" )+\';color:\'+lineCol+\';\'\n'
    '             +\'font-family:Courier New,monospace;white-space:nowrap;text-align:center;\'\n'
    '             +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '             +Math.round(t)+\'</div>\',\n'
    '        iconSize:[28,16], iconAnchor:[14,8], className:""\n'
    '      })\n'
    '    }).addTo(_synTmpLayer);\n'
    '  });\n'
    '  if (_synShowTmp) _synTmpLayer.addTo(MAP);\n'

    '  var _MAP2=window[Object.keys(window).filter(function(k){return k.startsWith("map_");})[0]];\n'
    '  if(_MAP2){\n'
    '    if(_synInstabLayer){_MAP2.removeLayer(_synInstabLayer);} _synInstabLayer=null; _synInstabLastKey=null;\n'
    '    if(_synThermalRidgeLayer){_MAP2.removeLayer(_synThermalRidgeLayer);} _synThermalRidgeLayer=null; _synThermalRidgeLastKey=null;\n'
    '    if(_synThermalTroughLayer){_MAP2.removeLayer(_synThermalTroughLayer);} _synThermalTroughLayer=null; _synThermalTroughLastKey=null;\n'
    '    if(_syn850MoistLayer){_MAP2.removeLayer(_syn850MoistLayer);} _syn850MoistLayer=null; _syn850MoistLastKey=null;\n'
    '    if(_synRidge700Layer){_MAP2.removeLayer(_synRidge700Layer);} _synRidge700Layer=null; _synRidge700LastKey=null;\n'
    '    if(_synTrough700Layer){_MAP2.removeLayer(_synTrough700Layer);} _synTrough700Layer=null; _synTrough700LastKey=null;\n'
    '    if(_synRidge500Layer){_MAP2.removeLayer(_synRidge500Layer);} _synRidge500Layer=null; _synRidge500LastKey=null;\n'
    '    if(_synTrough500Layer){_MAP2.removeLayer(_synTrough500Layer);} _synTrough500Layer=null; _synTrough500LastKey=null;\n'
    '  }\n'
    '  synRenderUA(ts);\n'
    '  synRender850Moist(ts);\n'
    '  synRenderInstab(ts);\n'
    '  synRenderThermalRidge(ts);\n'
    '  synRenderThermalTrough(ts);\n'
    '  if (_synShowRidge700)  _synToggleRT("_synRidge700Layer","_synShowRidge700","_synRidge700LastKey","btn-ridge700","thermal_ridge_700","#8B4513",true,ts);\n'
    '  if (_synShowTrough700) _synToggleRT("_synTrough700Layer","_synShowTrough700","_synTrough700LastKey","btn-trough700","thermal_trough_700","#8B4513",false,ts);\n'
    '  if (_synShowRidge500)  _synToggleRT("_synRidge500Layer","_synShowRidge500","_synRidge500LastKey","btn-ridge500","thermal_ridge_500","#0044cc",true,ts);\n'
    '  if (_synShowTrough500) _synToggleRT("_synTrough500Layer","_synShowTrough500","_synTrough500LastKey","btn-trough500","thermal_trough_500","#0044cc",false,ts);\n'
    # FIX 2: Re-sync all button colors to match actual show-state after every TS update
    '  if (_synShowStations)     _synBtnOn("btn-stns");     else _synBtnOff("btn-stns");\n'
    '  if (_synShowSlp)          _synBtnOn("btn-slp");      else _synBtnOff("btn-slp");\n'
    '  if (_synShowHL)           _synBtnOn("btn-hl");       else _synBtnOff("btn-hl");\n'
    '  if (_synShowTtd)          _synBtnOn("btn-ttd");      else _synBtnOff("btn-ttd");\n'
    '  if (_synShowTmp)          _synBtnOn("btn-tmp");      else _synBtnOff("btn-tmp");\n'
    '  if (_synShowSfcModel)     _synBtnOn("btn-sfcmod");   else _synBtnOff("btn-sfcmod");\n'
    '  if (_synShowTendRing)     _synBtnOn("btn-tend-ring"); else _synBtnOff("btn-tend-ring");\n'
    '  var _sfcSel=document.getElementById("sfc-tile-sel");\n'
    '  if(_sfcSel){\n'
    '    if(_synShowTmp) _sfcSel.value="tmp";\n'
    '    else if(_synShowTtd) _sfcSel.value="ttd";\n'
    '    else _sfcSel.value="";\n'
    '  }\n'
    '  if (_synShowUAStns)       _synBtnOn("btn-ua-stns");  else _synBtnOff("btn-ua-stns");\n'
    '  if (_synShowUAHght)       _synBtnOn("btn-ua-hght");  else _synBtnOff("btn-ua-hght");\n'
    '  if (_synShowUATemp)       _synBtnOn("btn-ua-temp");  else _synBtnOff("btn-ua-temp");\n'
    '  if (_synShowUATtdp)       _synBtnOn("btn-ua-ttdp");  else _synBtnOff("btn-ua-ttdp");\n'
    '  if (_synShowUASped)       _synBtnOn("btn-ua-sped");       else _synBtnOff("btn-ua-sped");\n'
    '  if (_synShowUATempBands)  _synBtnOn("btn-ua-tbands");     else _synBtnOff("btn-ua-tbands");\n'
    '  if (_synShow850Moist)     _synBtnOn("btn-850moist"); else _synBtnOff("btn-850moist");\n'
    '  if (_synShowInstab)       _synBtnOn("btn-instab");   else _synBtnOff("btn-instab");\n'
    '  if (_synShowThermalRidge) _synBtnOn("btn-thermal");  else _synBtnOff("btn-thermal");\n'
    '  if (_synShowThermalTrough)_synBtnOn("btn-trough");   else _synBtnOff("btn-trough");\n'
    '  if (_synShowRidge700)     _synBtnOn("btn-ridge700"); else _synBtnOff("btn-ridge700");\n'
    '  if (_synShowTrough700)    _synBtnOn("btn-trough700");else _synBtnOff("btn-trough700");\n'
    '  if (_synShowRidge500)     _synBtnOn("btn-ridge500"); else _synBtnOff("btn-ridge500");\n'
    '  if (_synShowTrough500)    _synBtnOn("btn-trough500");else _synBtnOff("btn-trough500");\n'
    '  var _activeMode = _synUALevel || (_synShowSlp||_synShowTmp||_synShowSfcModel ? "sfc" : "");\n'
    '  var _isAnalysis = (_synShowThermalRidge||_synShowTrough700||_synShowTrough500||_synShow850Moist||_synShowInstab);\n'
    '  if (_isAnalysis) _activeMode = "analysis";\n'
    '  ["sfc","850","700","500","250","analysis","drymicroburst"].forEach(function(m) {\n'
    '    var b = document.getElementById("btn-mode-"+m);\n'
    '    if (b) b.style.background = (m === mode) ? "#4a7fc1" : "#b0b8c8";\n'
    '  });\n'
    '}\n'

    # ── state vars ──
    'var _syn850MoistLayer    = null;\n'
    'var _synShow850Moist     = false;\n'
    'var _syn850MoistLastKey  = null;\n'
    'var _synInstabLayer      = null;\n'
    'var _synShowInstab       = false;\n'
    'var _synInstabLastKey    = null;\n'
    'var _synThermalRidgeLayer   = null;\n'
    'var _synShowThermalRidge    = false;\n'
    'var _synThermalRidgeLastKey = null;\n'
    'var _synThermalTroughLayer   = null;\n'
    'var _synShowThermalTrough    = false;\n'
    'var _synThermalTroughLastKey = null;\n'
    'var _synDtdxLayer        = null;\n'
    'var _synShowDtdx         = false;\n'
    'var _synRidge700Layer = null; var _synShowRidge700 = false; var _synRidge700LastKey = null;\n'
    'var _synTrough700Layer= null; var _synShowTrough700= false; var _synTrough700LastKey= null;\n'
    'var _synRidge500Layer = null; var _synShowRidge500 = false; var _synRidge500LastKey = null;\n'
    'var _synTrough500Layer= null; var _synShowTrough500= false; var _synTrough500LastKey= null;\n'
    'var _synShowSfcTtd       = false;\n'
    'var _synSfcTtdLayer      = null;\n'
    'var _synMoistRingLayer   = null;\n'
    'var _synTendRingLayer    = null;\n'
    'var _synShowTendRing     = false;\n'
    'var _synShowSlp          = false;\n'
    'var _synShowHL           = false;\n'
    'var _synShowTtd          = false;\n'
    'var _synShowTmp          = false;\n'
    'var _synShowSfcModel     = false;\n'
    'var _synSfcModelLayer    = null;\n'
    'var _synShowStations     = false;\n'
    'var _synHourFilter       = -1;\n'
    'var _synUAHourKey        = "0";\n'
    'var _synUALevel          = "";\n'
    'var _synUALayer          = null;\n'
    'var _synUAStnLayer       = null;\n'
    'var _synShowUAStns       = true;\n'
    'var _SYN_UA_STNS         = ' + _ts_ua_stn_json_str + ';\n'
    'var _synShowUAHght       = true;\n'
    'var _synShowUATemp       = true;\n'
    'var _synShowUAWC         = true;\n'   # follows Temp by default
    'var _synShowUATtdp       = false;\n'
    'var _synShowUASped       = false;\n'
    'var _synShowUATempBands  = ' + ('true' if UA_TEMP_BAND_SHOW else 'false') + ';\n'


    # ── toggle functions — color only, no textContent changes ──
    'function synToggleStations() {\n'
    '  _synShowStations = !_synShowStations;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowStations) { if (_synStnLayer) _synStnLayer.addTo(MAP); _synBtnOn("btn-stns"); }\n'
    '  else { if (_synStnLayer) MAP.removeLayer(_synStnLayer); _synBtnOff("btn-stns"); }\n'
    '}\n'

    'function synToggleLayer(which) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (which === "slp") {\n'
    '    _synShowSlp = !_synShowSlp; _synShowHL = _synShowSlp;\n'
    '    if (_synShowSlp) {\n'
    '      if (_synSlpLayer) _synSlpLayer.addTo(MAP);\n'
    '      if (_synHLLayer)  _synHLLayer.addTo(MAP);\n'
    '      _synBtnOn("btn-slp");\n'
    '    } else {\n'
    '      if (_synSlpLayer) MAP.removeLayer(_synSlpLayer);\n'
    '      if (_synHLLayer)  MAP.removeLayer(_synHLLayer);\n'
    '      _synBtnOff("btn-slp");\n'
    '    }\n'
    '  } else if (which === "hl") {\n'
    '    _synShowHL = !_synShowHL;\n'
    '    if (_synShowHL) { if (_synHLLayer) _synHLLayer.addTo(MAP); _synBtnOn("btn-hl"); }\n'
    '    else { if (_synHLLayer) MAP.removeLayer(_synHLLayer); _synBtnOff("btn-hl"); }\n'
    '  } else if (which === "ttd") {\n'
    '    _synShowTtd = !_synShowTtd;\n'
    '    _synShowSfcTtd = _synShowTtd;\n'
    '    if (_synShowTtd) {\n'
    '      if (_synTtdLayer) _synTtdLayer.addTo(MAP);\n'
    '      if (_synMoistRingLayer) _synMoistRingLayer.addTo(MAP);\n'
    '      _synBtnOn("btn-ttd"); _synBtnOn("btn-sfc-ttd");\n'
    '    } else {\n'
    '      if (_synTtdLayer) MAP.removeLayer(_synTtdLayer);\n'
    '      if (_synMoistRingLayer) MAP.removeLayer(_synMoistRingLayer);\n'
    '      _synBtnOff("btn-ttd"); _synBtnOff("btn-sfc-ttd");\n'
    '    }\n'
    '  } else if (which === "tmp") {\n'
    '    _synShowTmp = !_synShowTmp;\n'
    '    if (_synShowTmp) { if (_synTmpLayer) _synTmpLayer.addTo(MAP); _synBtnOn("btn-tmp"); }\n'
    '    else { if (_synTmpLayer) MAP.removeLayer(_synTmpLayer); _synBtnOff("btn-tmp"); }\n'
    '  }\n'
    '}\n'

    'function synToggleInstab() {\n'
    '  _synShowInstab = !_synShowInstab;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowInstab) { if (_synInstabLayer) _synInstabLayer.addTo(MAP); _synBtnOn("btn-instab"); }\n'
    '  else { if (_synInstabLayer) MAP.removeLayer(_synInstabLayer); _synBtnOff("btn-instab"); }\n'
    '}\n'

    'function synToggleThermalRidge() {\n'
    '  _synShowThermalRidge = !_synShowThermalRidge;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowThermalRidge) { if (_synThermalRidgeLayer) _synThermalRidgeLayer.addTo(MAP); _synBtnOn("btn-thermal"); }\n'
    '  else { if (_synThermalRidgeLayer) MAP.removeLayer(_synThermalRidgeLayer); _synBtnOff("btn-thermal"); }\n'
    '}\n'

    'function synToggleThermalTrough() {\n'
    '  _synShowThermalTrough = !_synShowThermalTrough;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowThermalTrough) { if (_synThermalTroughLayer) _synThermalTroughLayer.addTo(MAP); _synBtnOn("btn-trough"); }\n'
    '  else { if (_synThermalTroughLayer) MAP.removeLayer(_synThermalTroughLayer); _synBtnOff("btn-trough"); }\n'
    '}\n'

    'function synToggleSfcTtd() {\n'
'  synToggleLayer("ttd");\n'
'  var on = _synShowTtd;\n'
'  _synShowSfcTtd = on;\n'
'  if (on) _synBtnOn("btn-sfc-ttd"); else _synBtnOff("btn-sfc-ttd");\n'
'}\n'

    'function synToggleTendRing() {\n'
    '  _synShowTendRing = !_synShowTendRing;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowTendRing) { if (_synTendRingLayer) _synTendRingLayer.addTo(MAP); _synBtnOn("btn-tend-ring"); }\n'
    '  else { if (_synTendRingLayer) MAP.removeLayer(_synTendRingLayer); _synBtnOff("btn-tend-ring"); }\n'
    '}\n'
    'function synToggleSfcModel() {\n'
    '  _synShowSfcModel = !_synShowSfcModel;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowSfcModel) { if (_synSfcModelLayer) _synSfcModelLayer.addTo(MAP); _synBtnOn("btn-sfcmod");\n'
    '  } else { if (_synSfcModelLayer) MAP.removeLayer(_synSfcModelLayer); _synBtnOff("btn-sfcmod");\n'
    '  }\n'
    '}\n'
    '\n'
    'function synUpdateSfcTile(val) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  // turn off both first\n'
    '  _synShowTmp = false; _synShowTtd = false;\n'
    '  if (_synTmpLayer) MAP.removeLayer(_synTmpLayer);\n'
    '  if (_synTtdLayer) MAP.removeLayer(_synTtdLayer);\n'
    '  if (val === "tmp") { _synShowTmp = true; if (_synTmpLayer) _synTmpLayer.addTo(MAP); }\n'
    '  if (val === "ttd") { _synShowTtd = true; if (_synTtdLayer) _synTtdLayer.addTo(MAP); }\n'
    '}\n'

    'function synSfcMaster(val) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  var on = (val === "on");\n'
    '  _synShowSfcModel = on;\n'
    '  if (on) { if (_synSfcModelLayer) _synSfcModelLayer.addTo(MAP); _synBtnOn("btn-sfcmod"); }\n'
    '  else    { if (_synSfcModelLayer) MAP.removeLayer(_synSfcModelLayer); _synBtnOff("btn-sfcmod"); }\n'
    '  _synShowSlp = on; _synShowHL = on;\n'
    '  if (on) {\n'
    '    if (_synSlpLayer) _synSlpLayer.addTo(MAP);\n'
    '    if (_synHLLayer)  _synHLLayer.addTo(MAP);\n'
    '    _synBtnOn("btn-slp");\n'
    '  } else {\n'
    '    if (_synSlpLayer) MAP.removeLayer(_synSlpLayer);\n'
    '    if (_synHLLayer)  MAP.removeLayer(_synHLLayer);\n'
    '    _synBtnOff("btn-slp");\n'
    '  }\n'
    '  _synShowTmp = on;\n'
    '  if (on) { if (_synTmpLayer) _synTmpLayer.addTo(MAP); _synBtnOn("btn-tmp"); }\n'
    '  else    { if (_synTmpLayer) MAP.removeLayer(_synTmpLayer); _synBtnOff("btn-tmp"); }\n'
    '  _synShowSfcTtd = on; _synShowTtd = on;\n'
    '  if (on) {\n'
    '    if (_synTtdLayer) _synTtdLayer.addTo(MAP);\n'
    '    _synBtnOn("btn-sfc-ttd"); _synBtnOn("btn-ttd");\n'
    '  } else {\n'
    '    if (_synTtdLayer) MAP.removeLayer(_synTtdLayer);\n'
    '    _synBtnOff("btn-sfc-ttd"); _synBtnOff("btn-ttd");\n'
    '  }\n'
    '}\n'
    '\n'
    'function synToggle850Moist() {\n'
    '  _synShow850Moist = !_synShow850Moist;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShow850Moist) { if (_syn850MoistLayer) _syn850MoistLayer.addTo(MAP); _synBtnOn("btn-850moist"); }\n'
    '  else { if (_syn850MoistLayer) MAP.removeLayer(_syn850MoistLayer); _synBtnOff("btn-850moist"); }\n'
    '}\n'

    'function synToggleUAStns() {\n'
    '  _synShowUAStns = !_synShowUAStns;\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synShowUAStns) { if (_synUAStnLayer) _synUAStnLayer.addTo(MAP); _synBtnOn("btn-ua-stns"); }\n'
    '  else { if (_synUAStnLayer) MAP.removeLayer(_synUAStnLayer); _synBtnOff("btn-ua-stns"); }\n'
    '}\n'

    'function synToggleUA(field) {\n'
    '  var btnMap   = {"hght":"btn-ua-hght","temp":"btn-ua-temp","ttdp":"btn-ua-ttdp","sped":"btn-ua-sped"};\n'
    '  if (field==="hght")      _synShowUAHght = !_synShowUAHght;\n'
    '  else if (field==="temp") { _synShowUATemp = !_synShowUATemp; _synShowUAWC = _synShowUATemp; }\n'
    '  else if (field==="ttdp") _synShowUATtdp = !_synShowUATtdp;\n'
    '  else if (field==="sped") _synShowUASped = !_synShowUASped;\n'
    '  var show = (field==="hght"?_synShowUAHght:field==="temp"?_synShowUATemp:field==="ttdp"?_synShowUATtdp:_synShowUASped);\n'
    '  if (show) _synBtnOn(btnMap[field]); else _synBtnOff(btnMap[field]);\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  synRenderUA(sel ? sel.value : "");\n'
    '}\n'

    # ── 700/500 ridge/trough helpers ──
    'function synToggleUATempBands() {\n'
    '  _synShowUATempBands = !_synShowUATempBands;\n'
    '  if (_synShowUATempBands) _synBtnOn("btn-ua-tbands"); else _synBtnOff("btn-ua-tbands");\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  synRenderUA(sel ? sel.value : "");\n'
    '}\n'
    'function _synMakeRTLayer(uaKey, field, color, isRidge) {\n'
    '  var segs = (_SYN_UA[uaKey] || {})[field] || [];\n'
    '  var lg = L.layerGroup();\n'
    '  segs.forEach(function(seg) {\n'
    '    var ll = seg.coords.map(function(c){return [c[1],c[0]];});\n'
    '    if (isRidge) {\n'
    '      var _prev=null;\n'
    '      ll.forEach(function(pt){\n'
    '        if(_prev===null||Math.abs(pt[0]-_prev[0])+Math.abs(pt[1]-_prev[1])>=1.5){\n'
    '          L.circleMarker(pt,{radius:4,color:color,weight:1.5,fillColor:color,fillOpacity:0.7})\n'
    '           .bindTooltip(field).addTo(lg);\n'
    '          _prev=pt;\n'
    '        }\n'
    '      });\n'
    '    } else {\n'
    '      var _prevT=null;\n'
    '      ll.forEach(function(pt){\n'
    '        if(_prevT===null||Math.abs(pt[0]-_prevT[0])+Math.abs(pt[1]-_prevT[1])>=1.5){\n'
    '          L.marker(pt,{icon:L.divIcon({\n'
    '            html:\'<div style="width:0;height:0;\'\n'
    '                +\'border-left:6px solid transparent;\'\n'
    '                +\'border-right:6px solid transparent;\'\n'
    '                +\'border-bottom:11px solid \'+color+\';\'\n'
    '                +\'opacity:0.85;"></div>\',\n'
    '            iconSize:[12,11],iconAnchor:[6,11],className:""})}).addTo(lg);\n'
    '          _prevT=pt;\n'
    '        }\n'
    '      });\n'
    '    }\n'
    '    L.marker([seg.label_lat,seg.label_lon],{icon:L.divIcon({\n'
    '      html:\'<div style="font-size:10px;font-weight:bold;color:\'+color+\';\'\n'
    '          +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '          +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '          +{"thermal_ridge_700":"700Ridge","thermal_trough_700":"700Trough","thermal_ridge_500":"500Ridge","thermal_trough_500":"500Trough"}[field]+\'</div>\',\n'
    '      iconSize:[70,14],iconAnchor:[35,7],className:""})}).addTo(lg);\n'
    '  });\n'
    '  return lg;\n'
    '}\n'
    'function _synToggleRT(layerVar,showVar,lastKeyVar,btnId,field,color,isRidge,ts){\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if(!keys.length)return;\n'
    '  var MAP=window[keys[0]];\n'
    '  var uaKey=_synUAHourKey;\n'
    '  if(window[showVar]){\n'
    '    if(!window[layerVar]||window[lastKeyVar]!==uaKey){\n'
    '      if(window[layerVar])MAP.removeLayer(window[layerVar]);\n'
    '      window[layerVar]=_synMakeRTLayer(uaKey,field,color,isRidge);\n'
    '      window[lastKeyVar]=uaKey;\n'
    '    }\n'
    '    window[layerVar].addTo(MAP);\n'
    '    _synBtnOn(btnId);\n'
    '  } else {\n'
    '    if(window[layerVar])MAP.removeLayer(window[layerVar]);\n'
    '    _synBtnOff(btnId);\n'
    '  }\n'
    '}\n'
    'function synToggleRidge700(){\n'
    '  _synShowRidge700=!_synShowRidge700;\n'
    '  var sel=document.getElementById("ts-select");\n'
    '  _synToggleRT("_synRidge700Layer","_synShowRidge700","_synRidge700LastKey","btn-ridge700","thermal_ridge_700","#8B4513",true,sel?sel.value:"");\n'
    '}\n'
    'function synToggleTrough700(){\n'
    '  _synShowTrough700=!_synShowTrough700;\n'
    '  var sel=document.getElementById("ts-select");\n'
    '  _synToggleRT("_synTrough700Layer","_synShowTrough700","_synTrough700LastKey","btn-trough700","thermal_trough_700","#8B4513",false,sel?sel.value:"");\n'
    '}\n'
    'function synToggleRidge500(){\n'
    '  _synShowRidge500=!_synShowRidge500;\n'
    '  var sel=document.getElementById("ts-select");\n'
    '  _synToggleRT("_synRidge500Layer","_synShowRidge500","_synRidge500LastKey","btn-ridge500","thermal_ridge_500","#0044cc",true,sel?sel.value:"");\n'
    '}\n'
    'function synToggleTrough500(){\n'
    '  _synShowTrough500=!_synShowTrough500;\n'
    '  var sel=document.getElementById("ts-select");\n'
    '  _synToggleRT("_synTrough500Layer","_synShowTrough500","_synTrough500LastKey","btn-trough500","thermal_trough_500","#0044cc",false,sel?sel.value:"");\n'
    '}\n'

    'function synToggleDtdx() {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synDtdxLayer && MAP.hasLayer(_synDtdxLayer)) {\n'
    '    MAP.removeLayer(_synDtdxLayer);\n'
    '    _synBtnOff("btn-dtdx");\n'
    '  } else {\n'
    '    var sel = document.getElementById("ts-select");\n'
    '    var ts = sel ? sel.value : "";\n'
    '    var _mh = ts.match(/^\\d{2}(\\d{2})\\d{2}Z$/);\n'
    '    var tsHour = _mh ? parseInt(_mh[1], 10) : 0;\n'
    '    var uaKey  = (tsHour >= 12 && tsHour < 18) ? "12" : "0";\n'
    '    var pts = (_SYN_UA[uaKey] || {}).dtdx_zero_pts || [];\n'
    '    if (_synDtdxLayer) MAP.removeLayer(_synDtdxLayer);\n'
    '    _synDtdxLayer = L.layerGroup();\n'
    '    pts.forEach(function(p) {\n'
    '      L.circleMarker([p.lat, p.lon], {\n'
    '        radius:3, color:"#7700bb", weight:1, opacity:0.85,\n'
    '        fillColor:"#aa44dd", fillOpacity:0.6\n'
    '      }).addTo(_synDtdxLayer);\n'
    '    });\n'
    '    _synDtdxLayer.addTo(MAP);\n'
    '    _synBtnOn("btn-dtdx");\n'
    '  }\n'
    '}\n'

    # ── render functions ──
    'function synRenderInstab(ts) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!ts) return;\n'
    '  var uaKey = _synUAHourKey;\n'
    '  if (uaKey === _synInstabLastKey && _synInstabLayer) {\n'
    '    if (_synShowInstab) _synInstabLayer.addTo(MAP);\n'
    '    return;\n'
    '  }\n'
    '  _synInstabLastKey = uaKey;\n'
    '  if (_synInstabLayer) { MAP.removeLayer(_synInstabLayer); _synInstabLayer = null; }\n'
    '  var instabContours = ((_SYN_UA[uaKey] || {}).instab) || [];\n'
    '  _synInstabLayer = L.layerGroup();\n'
    '  [16, 18].forEach(function(bandLvl) {\n'
    '    var isCB      = (bandLvl === 18);\n'
    '    var fillCol   = isCB ? "#ff4400" : "#ffbb77";\n'
    '    var borderCol = isCB ? "#cc2200" : "#cc6600";\n'
    '    var label     = isCB ? "CB" : "TCU";\n'
    '    instabContours.filter(function(ct){ return Math.round(ct.level) === bandLvl; })\n'
    '    .forEach(function(ct) {\n'
    '      var ll = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '      var _instabS = window._SYN_STYLE ? window._SYN_STYLE.instab : {weight:1.2,opacity:0.85,tcuFill:"#ffbb77",tcuBorder:"#cc6600",tcuFillOpacity:0.35,cbFill:"#ff4400",cbBorder:"#cc2200",cbFillOpacity:0.55};\n'
'      var fillCol   = isCB ? _instabS.cbFill   : _instabS.tcuFill;\n'
'      var borderCol = isCB ? _instabS.cbBorder  : _instabS.tcuBorder;\n'
'      L.polygon(ll, {\n'
'        color:borderCol, weight:_instabS.weight, opacity:_instabS.opacity,\n'
'        fillColor:fillCol, fillOpacity:isCB?_instabS.cbFillOpacity:_instabS.tcuFillOpacity, dashArray:isCB?null:"5 3"\n'
    '      }).bindTooltip("T700-500 "+(isCB?"\\u226518":"16\\u201318")+"\\u00b0C ("+label+")")\n'
    '        .addTo(_synInstabLayer);\n'
    '      L.marker([ct.label_lat, ct.label_lon], {\n'
    '        icon: L.divIcon({\n'
    '          html: \'<div style="font-size:13px;font-weight:bold;color:\'+borderCol+\';\'\n'
    '              +\'font-family:Arial Black,sans-serif;\'\n'
    '              +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '              + label + \'</div>\',\n'
    '          iconSize:[40,18], iconAnchor:[20,9], className:""\n'
    '        })\n'
    '      }).addTo(_synInstabLayer);\n'
    '    });\n'
    '  });\n'
    '  if (_synShowInstab) _synInstabLayer.addTo(MAP);\n'
    '}\n'

    'function synRenderThermalRidge(ts) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!ts) return;\n'
    '  var uaKey = _synUAHourKey;\n'
    '  if (uaKey === _synThermalRidgeLastKey && _synThermalRidgeLayer) {\n'
    '    if (_synShowThermalRidge) _synThermalRidgeLayer.addTo(MAP);\n'
    '    return;\n'
    '  }\n'
    '  _synThermalRidgeLastKey = uaKey;\n'
    '  if (_synThermalRidgeLayer) { MAP.removeLayer(_synThermalRidgeLayer); _synThermalRidgeLayer = null; }\n'
    '  var ridgeSegs = (_SYN_UA[uaKey] || {}).thermal_ridge_850 || [];\n'
    '  _synThermalRidgeLayer = L.layerGroup();\n'
    '  ridgeSegs.forEach(function(seg) {\n'
    '    var ll = seg.coords.map(function(c){return [c[1],c[0]];});\n'
    '    var _prev = null;\n'
    '    ll.forEach(function(pt) {\n'
    '      if (_prev === null || Math.abs(pt[0]-_prev[0]) + Math.abs(pt[1]-_prev[1]) >= 1.5) {\n'
    '        L.circleMarker(pt, {radius:4, color:"#cc0000", weight:1.5, fillColor:"#cc0000", fillOpacity:0.7})\n'
    '         .bindTooltip("850 Ridge").addTo(_synThermalRidgeLayer);\n'
    '        _prev = pt;\n'
    '      }\n'
    '    });\n'
    '    L.marker([seg.label_lat, seg.label_lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: \'<div style="font-size:10px;font-weight:bold;color:#cc0000;\'\n'
    '            +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '            +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">850Ridge</div>\',\n'
    '        iconSize:[60,14], iconAnchor:[30,7], className:""\n'
    '      })\n'
    '    }).addTo(_synThermalRidgeLayer);\n'
    '  });\n'
    '  if (_synShowThermalRidge) _synThermalRidgeLayer.addTo(MAP);\n'
    '}\n'

    'function synRenderThermalTrough(ts) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!ts) return;\n'
    '  var uaKey = _synUAHourKey;\n'
    '  if (uaKey === _synThermalTroughLastKey && _synThermalTroughLayer) {\n'
    '    if (_synShowThermalTrough) _synThermalTroughLayer.addTo(MAP);\n'
    '    return;\n'
    '  }\n'
    '  _synThermalTroughLastKey = uaKey;\n'
    '  if (_synThermalTroughLayer) { MAP.removeLayer(_synThermalTroughLayer); _synThermalTroughLayer = null; }\n'
    '  var troughSegs = (_SYN_UA[uaKey] || {}).thermal_trough_850 || [];\n'
    '  _synThermalTroughLayer = L.layerGroup();\n'
    '  troughSegs.forEach(function(seg) {\n'
    '    var _prevT = null;\n'
    '    seg.coords.forEach(function(c) {\n'
    '      var pt = [c[1], c[0]];\n'
    '      if (_prevT === null || Math.abs(pt[0]-_prevT[0]) + Math.abs(pt[1]-_prevT[1]) >= 1.5) {\n'
    '        L.marker(pt, {\n'
    '          icon: L.divIcon({\n'
    '            html: \'<div style="width:0;height:0;\'\n'
    '                +\'border-left:6px solid transparent;\'\n'
    '                +\'border-right:6px solid transparent;\'\n'
    '                +\'border-bottom:11px solid #cc0000;\'\n'
    '                +\'opacity:0.85;"></div>\',\n'
    '            iconSize:[12,11], iconAnchor:[6,11], className:""\n'
    '          })\n'
    '        }).addTo(_synThermalTroughLayer);\n'
    '        _prevT = pt;\n'
    '      }\n'
    '    });\n'
    '    L.marker([seg.label_lat, seg.label_lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: \'<div style="font-size:10px;font-weight:bold;color:#cc0000;\'\n'
    '            +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '            +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">850Trough</div>\',\n'
    '        iconSize:[65,14], iconAnchor:[32,7], className:""\n'
    '      })\n'
    '    }).addTo(_synThermalTroughLayer);\n'
    '  });\n'
    '  if (_synShowThermalTrough) _synThermalTroughLayer.addTo(MAP);\n'
    '}\n'

    # FIX 3: synRender850Moist — removed unconditional removeLayer at top;
    # layer is only removed inside the rebuild block, preserving visibility when re-rendering same UA key
    'function synRender850Moist(ts) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!ts) return;\n'
    '  var uaKey = _synUAHourKey;\n'
    '  if (uaKey === _syn850MoistLastKey && _syn850MoistLayer) {\n'
    '    if (_synShow850Moist) _syn850MoistLayer.addTo(MAP);\n'
    '    return;\n'
    '  }\n'
    '  _syn850MoistLastKey = uaKey;\n'
    '  if (_syn850MoistLayer) { MAP.removeLayer(_syn850MoistLayer); _syn850MoistLayer = null; }\n'
    '  var m850 = (_SYN_UA[uaKey] || {levels:{}}).levels["850"] || {};\n'
    '  _syn850MoistLayer = L.layerGroup();\n'
    '  (m850.ttdp || []).forEach(function(ct) {\n'
    '    if (ct.level > 2) return;\n'
    '    var _ll850 = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    L.polygon(_ll850, {color:"#cc0000", weight:1.5, opacity:0.9, dashArray:"6 4", fillColor:"#add8e6", fillOpacity:0.4})\n'
    '     .bindTooltip("850 hPa T-Td="+Math.round(ct.level)+"°C").addTo(_syn850MoistLayer);\n'
    '    var _mid850 = _ll850[Math.floor(_ll850.length/2)];\n'
    '    L.marker([_mid850[0], _mid850[1]], {\n'
    '      icon: L.divIcon({\n'
    '        html: \'<div style="font-size:11px;font-weight:bold;color:#cc0000;\'\n'
    '            +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '            +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">850 moisture</div>\',\n'
    '        iconSize:[90,14], iconAnchor:[45,7], className:""\n'
    '      })\n'
    '    }).addTo(_syn850MoistLayer);\n'
    '  });\n'
    '  if (_synShow850Moist) _syn850MoistLayer.addTo(MAP);\n'
    '}\n'

    'function synUpdateUALevel(lvl) {\n'
    '  _synUALevel = lvl;\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  synRenderUA(sel ? sel.value : "");\n'
    '}\n'

    'var _uaBaseCol = "#1a5080";\n'
    'function synRenderUA(ts) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (_synUALayer) { MAP.removeLayer(_synUALayer); _synUALayer = null; }\n'
    '  if (!ts) return;\n'
    '  if (!_synUALevel) {\n'
    '    if (_synUAStnLayer) { MAP.removeLayer(_synUAStnLayer); _synUAStnLayer = null; }\n'
    '    var uaStnList0 = _SYN_UA_STNS[_synUAHourKey] || [];\n'
    '    _synUAStnLayer = L.layerGroup();\n'
    '    uaStnList0.forEach(function(d) {\n'
    '      L.marker([d.lat, d.lon], {\n'
    '        icon: L.divIcon({\n'
    '          html: \'<div style="width:8px;height:8px;background:#ff8800;border-radius:50%;border:1px solid #cc6600;"></div>\',\n'
    '          iconSize:[8,8], iconAnchor:[4,4], className:""\n'
    '        }), zIndexOffset:100\n'
    '      }).bindPopup(d.popup,{maxWidth:320,closeButton:true}).bindTooltip(d.tip).addTo(_synUAStnLayer);\n'
    '    });\n'
    '    if (_synShowUAStns) _synUAStnLayer.addTo(MAP);\n'
    '    return;\n'
    '  }\n'
    '  var uaKey  = _synUAHourKey;\n'
    '  var uaData = (_SYN_UA[uaKey] || {levels:{}}).levels[_synUALevel] || {};\n'
    '  _synUALayer = L.layerGroup();\n'
    '  var _uaS = window._SYN_STYLE ? window._SYN_STYLE.ua : {color:"#1a5080",weight:1.4,boldWeight:2.8,opacity:0.85,labelSize:"10px"};\n'
    '  if (_synShowUAHght) {\n'
    '    (uaData.hght || []).forEach(function(ct) {\n'
    '      var ll = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '      var _isKey = (KEY_HGT_DAM[_synUALevel] && (Math.round(ct.level)===KEY_HGT_DAM[_synUALevel]||Math.round(ct.level)===KEY_HGT_M[_synUALevel]));\n'
    '      L.polyline(ll, {color:_uaS.color, weight:_isKey?_uaS.boldWeight:_uaS.weight, opacity:_isKey?1.0:_uaS.opacity})\n'
    '       .bindTooltip(_synUALevel+" hPa Hgt="+Math.round(ct.level)+"dam").addTo(_synUALayer);\n'
    '      var hgtInterval = (_synUALevel==="850")?30:(_synUALevel==="700")?60:(_synUALevel==="500")?60:120;\n'
    '      var hgtAnchor   = (_synUALevel==="850")?1140:(_synUALevel==="700")?2520:(_synUALevel==="500")?4800:9600;\n'
    '      var hgtRem = Math.round(ct.level - hgtAnchor);\n'
    '      if (hgtRem >= 0 && hgtRem % hgtInterval < 1) {\n'
    '        var _skLat = 54.0, _skLon = -106.0;\n'
    '        var _best = null, _bestDist = 1e9;\n'
    '        ct.coords.forEach(function(c){\n'
    '          var dlat=c[1]-_skLat, dlon=c[0]-_skLon;\n'
    '          var d=dlat*dlat+dlon*dlon;\n'
    '          if(d<_bestDist){_bestDist=d;_best=c;}\n'
    '        });\n'
    '        var _lblLat = _best ? _best[1] : ct.label_lat;\n'
    '        var _lblLon = _best ? _best[0] : ct.label_lon;\n'
    '        L.marker([_lblLat,_lblLon],{icon:L.divIcon({\n'
    '        html:\'<div style="font-size:\'+_uaS.labelSize+\';font-weight:bold;color:#fff;\'\n'
    '              +\'font-family:Courier New,monospace;background:\'+_uaS.color+\';\'\n'
    '              +\'padding:0 3px;line-height:1.4;text-align:center;min-width:28px;">\'\n'
    '              +Math.round(ct.level/10)+\'</div>\',\n'
    '          iconSize:[32,14],iconAnchor:[16,7],className:""})}).addTo(_synUALayer);\n'
    '      }\n'
    '    });\n'
    '  }\n'
    '  var _uaTmpS = window._SYN_STYLE ? window._SYN_STYLE.uatemp : {weight:1.0,boldWeight:1.4,opacity:0.75,labelSize:"8px"};\n'
    '  if (_synShowUATemp) {\n'
    '    (uaData.temp || []).forEach(function(ct) {\n'
    '      var t = ct.level;\n'
    '      var col = t>0?"rgb("+Math.round(180+75*Math.min(t/40,1))+",0,0)"\n'
    '              : t<0?"rgb(0,0,"+Math.round(180+75*Math.min(Math.abs(t)/40,1))+")":"#00bb00";\n'
    '      var ll = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '      var isBold = (Math.round(t) % 10 === 0);\n'
    '      L.polyline(ll,{color:col,weight:isBold?_uaTmpS.boldWeight:_uaTmpS.weight,opacity:isBold?_uaTmpS.opacity:_uaTmpS.opacity*0.8,dashArray:"6 4"})\n'
    '       .bindTooltip(_synUALevel+" hPa T="+t.toFixed(1)+"°C").addTo(_synUALayer);\n'
    '      var _bcLat = 54.0, _bcLon = -130.0;\n'
    '      var _bcBest = null, _bcBestDist = 1e9;\n'
    '      ct.coords.forEach(function(c){\n'
    '        var dlat=c[1]-_bcLat, dlon=c[0]-_bcLon;\n'
    '        var d=dlat*dlat+dlon*dlon;\n'
    '        if(d<_bcBestDist){_bcBestDist=d;_bcBest=c;}\n'
    '      });\n'
    '      var _bcLblLat = _bcBest ? _bcBest[1] : ct.label_lat;\n'
    '      var _bcLblLon = _bcBest ? _bcBest[0] : ct.label_lon;\n'
    '      var _tVal = Math.round(t);\n'
    '      var _tBg  = _tVal > 0 ? "#cc0000" : _tVal < 0 ? "#0044cc" : "#008800";\n'
    '      L.marker([_bcLblLat, _bcLblLon], { icon: L.divIcon({\n'
    '        html: \'<div style="font-size:12px;font-weight:\' + (isBold ? "900" : "bold") + \';\'\n'
    '            + \'color:#ffffff;background:transparent;\'\n'
    '            + \'font-family:Courier New,monospace;\'\n'
    '            + \'text-shadow:-2px -2px 0 \' + _tBg + \',2px -2px 0 \' + _tBg + \',-2px 2px 0 \' + _tBg + \',2px 2px 0 \' + _tBg + \',\'\n'
    '            + \'-2px 0 0 \' + _tBg + \',2px 0 0 \' + _tBg + \',0 -2px 0 \' + _tBg + \',0 2px 0 \' + _tBg + \';\'\n'
    '            + \'padding:0 3px;line-height:1.4;text-align:center;width:28px;box-sizing:border-box;">\'\n'
    '            + _tVal + \'</div>\',\n'
    '        iconSize: [28,14], iconAnchor: [14,7], className: ""\n'
    '      }) }).addTo(_synUALayer);\n'
    '    });\n'
    '  }\n'
    '  if (_synShowUATemp && _synShowUATempBands) {\n'
    '    var _tbFills = uaData.temp_band_fills || [];\n'
    '    var _tbOpacity = (window._SYN_STYLE&&window._SYN_STYLE.uatempbands) ? window._SYN_STYLE.uatempbands.opacity : 0.25;\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    var _tbMap = keys.length ? window[keys[0]] : null;\n'
    '    if (_tbMap && !_tbMap.getPane("tempbandsPane")) {\n'
    '      _tbMap.createPane("tempbandsPane");\n'
    '      _tbMap.getPane("tempbandsPane").style.zIndex = 200;\n'
    '      _tbMap.getPane("tempbandsPane").style.pointerEvents = "none";\n'
    '    }\n'
    '    _tbFills.forEach(function(poly) {\n'
    '      if (!poly.coords || poly.coords.length < 3) return;\n'
    '      if (poly.color === "#ffffff") return;\n'
    '      var outerLL = poly.coords.map(function(c){return [c[1],c[0]];});\n'
    '      var holes = (poly.holes || []).map(function(hole) {\n'
    '        return hole.map(function(c){return [c[1],c[0]];});\n'
    '      });\n'
    '      var rings = [outerLL].concat(holes);\n'
    '      L.polygon(rings,{color:"none",weight:0,fillColor:poly.color,fillOpacity:_tbOpacity,fillRule:"evenodd",interactive:false,pane:"tempbandsPane"})\n'
    '       .addTo(_synUALayer);\n'
    '    });\n'
    '  }\n'
    '  if (_synShowUATtdp) {\n'
    '    (uaData.ttdp || []).forEach(function(ct) {\n'
    '      var ll = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '      var _uaTtdS = window._SYN_STYLE ? window._SYN_STYLE.uattd : {weight:1.1,opacity:0.65,labelSize:"8px"};\n'
    '      L.polyline(ll,{color:"#00aa44",weight:_uaTtdS.weight,opacity:_uaTtdS.opacity,dashArray:"4 4"})\n'
    '       .bindTooltip(_synUALevel+" hPa T-Td="+Math.round(ct.level)+"°C").addTo(_synUALayer);\n'
    '      if (Math.round(ct.level) % 4 === 0) {\n'
    '        L.marker([ct.label_lat,ct.label_lon],{icon:L.divIcon({\n'
    '          html:\'<div style="font-size:\'+_uaS.labelSize+\';font-weight:bold;color:#00aa44;\'\n'
    '              +\'font-family:Courier New,monospace;\'\n'
    '              +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '              +Math.round(ct.level)+\'</div>\',\n'
    '          iconSize:[28,14],iconAnchor:[14,7],className:""})}).addTo(_synUALayer);\n'
    '      }\n'
    '    });\n'
    '  }\n'
    '  if (_synShowUASped) {\n'
    '    (uaData.sped || []).forEach(function(ct) {\n'
    '      var ll   = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '      var frac = Math.min(ct.level/60,1);\n'
    '      var _uaSpedS = window._SYN_STYLE ? window._SYN_STYLE.uasped : {weightMin:0.8,weightMax:2.0,opacityMin:0.45,opacityMax:0.90};\n'
    '      L.polyline(ll,{color:_uaBaseCol,weight:_uaSpedS.weightMin+(_uaSpedS.weightMax-_uaSpedS.weightMin)*frac,opacity:_uaSpedS.opacityMin+(_uaSpedS.opacityMax-_uaSpedS.opacityMin)*frac,dashArray:"3 3"})\n'
    '       .bindTooltip(_synUALevel+" hPa "+Math.round(ct.level)+"m/s").addTo(_synUALayer);\n'
    '      if (Math.round(ct.level) % 10 === 0) {\n'
    '        L.marker([ct.label_lat,ct.label_lon],{icon:L.divIcon({\n'
    '          html:\'<div style="font-size:\'+_uaS.labelSize+\';font-weight:bold;color:\'+_uaBaseCol+\';\'\n'
    '              +\'font-family:Courier New,monospace;\'\n'
    '              +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '              +Math.round(ct.level)+\'</div>\',\n'
    '          iconSize:[28,14],iconAnchor:[14,7],className:""})}).addTo(_synUALayer);\n'
    '      }\n'
    '    });\n'
    '  }\n'
    '  // ── UA H/L centres for the selected level ──────────────────────────────\n'
    '  if (_synShowUAHght && _synUALevel) {\n'
    '    var _uaHL = (_SYN_UA[uaKey] || {})["hl_" + _synUALevel] || [];\n'
    '    _uaHL.forEach(function(c) {\n'
    '      var _shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";\n'
    '      var _html = \'<div style="font-size:61px;font-weight:bold;color:#000000;\'\n'
    '        + \'font-family:Palatino Linotype,Palatino,serif;line-height:1;\'\n'
    '        + \'text-shadow:\' + _shadow + \';pointer-events:none;">\' + c.type + \'</div>\';\n'
    '      L.marker([c.lat, c.lon], {\n'
    '        icon: L.divIcon({html: _html, iconSize:[70,65], iconAnchor:[35,32], className:""}),\n'
    '        zIndexOffset: 200\n'
    '      }).bindTooltip(_synUALevel + " hPa " + c.type).addTo(_synUALayer);\n'
    '    });\n'
    '  }\n'
       # ── ADD THIS: W/C thermal centres ─────────────────────────────────────
    '  if (_synShowUATemp && _synUALevel) {\n'
    '    var _uaWC = (((_SYN_UA[uaKey] || {}).levels || {})[_synUALevel] || {}).wc || [];\n'
    '    _uaWC.forEach(function(c) {\n'
    '      var isW   = c.type === "W";\n'
    '      var _col  = isW ? "#cc2200" : "#0055cc";\n'
    '      var _shad = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";\n'
    '      var _html = \'<div style="font-size:48px;font-weight:bold;color:\' + _col + \';\'\n'
    '        + \'font-family:Palatino Linotype,Palatino,serif;line-height:1;\'\n'
    '        + \'text-shadow:\' + _shad + \';">\' + c.type + \'</div>\';\n'
    '      L.marker([c.lat, c.lon], {\n'
    '        icon: L.divIcon({html:_html, iconSize:[56,52], iconAnchor:[28,26], className:""}),\n'
    '        zIndexOffset: 190\n'
    '      }).bindTooltip(_synUALevel + " hPa " + (isW?"Warm":"Cold") + " " + c.val.toFixed(1) + "°C")\n'
    '        .addTo(_synUALayer);\n'
    '    });\n'
    '  }\n'
    '  console.log("UA H/L debug: uaKey="+uaKey+" level="+_synUALevel+" showHght="+_synShowUAHght);\n'
    '  var _hlDebug = (_SYN_UA[uaKey] || {})["hl_" + _synUALevel] || [];\n'
    '  console.log("UA H/L entries:", _hlDebug.length, JSON.stringify(_hlDebug.slice(0,3)));\n'
    '  _synUALayer.addTo(MAP);\n'
    '  if (_synUAStnLayer) { MAP.removeLayer(_synUAStnLayer); _synUAStnLayer = null; }\n'
    '  var uaStnList = _SYN_UA_STNS[uaKey] || [];\n'
    '  _synUAStnLayer = L.layerGroup();\n'
    '  uaStnList.forEach(function(d) {\n'
    '    var svgInfo = (d.svgs && _synUALevel) ? d.svgs[_synUALevel] : null;\n'
    '    var icon;\n'
    '    if (svgInfo) {\n'
    '      icon = L.divIcon({html:svgInfo.svg, iconSize:[svgInfo.w,svgInfo.h],\n'
    '        iconAnchor:[Math.round(svgInfo.w/2),Math.round(svgInfo.h/2)], className:""});\n'
    '    } else {\n'
    '      icon = L.divIcon({\n'
    '        html:\'<div style="width:8px;height:8px;background:#ff8800;border-radius:50%;border:1px solid #cc6600;"></div>\',\n'
    '        iconSize:[8,8], iconAnchor:[4,4], className:""});\n'
    '    }\n'
    '    L.marker([d.lat,d.lon],{icon:icon,zIndexOffset:100})\n'
    '     .bindPopup(d.popup,{maxWidth:320,closeButton:true}).bindTooltip(d.tip).addTo(_synUAStnLayer);\n'
    '  });\n'
    '  if (_synShowUAStns) _synUAStnLayer.addTo(MAP);\n'
    '}\n'

    'function synFilterHour(h) {\n'
    '  _synHourFilter = h;\n'
    '  _synUAHourKey  = (h===12) ? "12" : "0";\n'
    '  _synBtnOff("btn-f00"); _synBtnOff("btn-f12");\n'
    '  _synBtnOn(h===0?"btn-f00":"btn-f12");\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  if (!sel) return;\n'
    '  var filtered = _SYN_TS_LIST.filter(function(ts) {\n'
    '    if (h===-1) return true;\n'
    '    var _mf = ts.match(/^\\d{2}(\\d{2})\\d{2}Z$/);\n'
    '    var hour = _mf ? parseInt(_mf[1],10) : -1;\n'
    '    return (h===0) ? (hour>=0&&hour<6) : (hour>=12&&hour<18);\n'
    '  });\n'
    '  var _noMatch = !filtered.length;\n'
    '  if (_noMatch) filtered = _SYN_TS_LIST;\n'
    '  var bestMatch = null;\n'
    '  for (var _bi=filtered.length-1; _bi>=0; _bi--) {\n'
    '    var _bm=filtered[_bi].match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
    '    if (_bm && parseInt(_bm[2],10)===h && parseInt(_bm[3],10)===0){bestMatch=filtered[_bi];break;}\n'
    '  }\n'
    '  if (!bestMatch) bestMatch = filtered[0];\n'
    '  if (!bestMatch) bestMatch = filtered[filtered.length-1];\n'
    '  if (!_noMatch && bestMatch) {\n'
    '    var _chk=bestMatch.match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
    '    if (_chk && !(parseInt(_chk[2],10)===h && parseInt(_chk[3],10)===0)) _noMatch=true;\n'
    '  }\n'
    '  sel.value = bestMatch;\n'
    '  var _nmEl = document.getElementById("syn-no-match-msg");\n'
    '  if (_nmEl) {\n'
    '    if (_noMatch) {\n'
    '      var _bm2=bestMatch.match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
    '      var _bmLabel=_bm2?(parseInt(_bm2[1])+"d "+_bm2[2]+"Z"):bestMatch;\n'
    '      _nmEl.textContent="\\u26A0 No matched METAR for "+(h===0?"00":"12")+"Z — Showing "+_bmLabel;\n'
    '      _nmEl.style.display="block";\n'
    '    } else { _nmEl.style.display="none"; }\n'
    '  }\n'
    '  synUpdateTS(bestMatch);\n'
    '}\n'

    'function synInitDropdown() {\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  if (!sel) { setTimeout(synInitDropdown,200); return; }\n'
    '  _SYN_TS_LIST.forEach(function(ts) {\n'
    '    var opt = document.createElement("option");\n'
    '    opt.value = ts;\n'
    '    var mts = ts.match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
    '    opt.textContent = mts ? (parseInt(mts[1])+"d "+mts[2]+"Z") : ts;\n'
    '    sel.appendChild(opt);\n'
    '  });\n'
    '  var _uaKeysSorted = Object.keys(_SYN_UA_STNS).map(function(k){return parseInt(k,10);}).sort(function(a,b){return b-a;});\n'
    '  var _targetHour = _uaKeysSorted.length ? _uaKeysSorted[0] : -1;\n'
    '  var latest = null;\n'
    '  for (var _i=_SYN_TS_LIST.length-1; _i>=0; _i--) {\n'
    '    var _tm=_SYN_TS_LIST[_i].match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
    '    if (!_tm) continue;\n'
    '    if (parseInt(_tm[2],10)===_targetHour && parseInt(_tm[3],10)===0){latest=_SYN_TS_LIST[_i];break;}\n'
    '  }\n'
    '  if (!latest) {\n'
    '    for (var _j=_SYN_TS_LIST.length-1; _j>=0; _j--) {\n'
    '      var _tm2=_SYN_TS_LIST[_j].match(/^\\d{2}(\\d{2})\\d{2}Z$/);\n'
    '      if (!_tm2) continue;\n'
    '      var _th2=parseInt(_tm2[1],10);\n'
    '      var _win=(_targetHour===12)?(_th2>=12&&_th2<18):(_th2>=0&&_th2<6);\n'
    '      if (_win){latest=_SYN_TS_LIST[_j];break;}\n'
    '    }\n'
    '  }\n'
    '  if (!latest) latest = _SYN_TS_LIST[_SYN_TS_LIST.length-1];\n'
    '  var _months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];\n'
'  [0,12].forEach(function(h) {\n'
'    var btnId = h===0?"btn-f00":"btn-f12";\n'
'    var btn = document.getElementById(btnId);\n'
'    if (!btn) return;\n'
'    var _key = h===0?"0":"12";\n'
'    var _dt = (_SYN_UA_DATES||{})[_key];\n'
'    if (_dt) {\n'
'      var _dp=_dt.match(/(\\d{4})-(\\d{2})-(\\d{2})/);\n'
'      if (_dp) {\n'
'        btn.textContent=_months[parseInt(_dp[2],10)-1]+" "+parseInt(_dp[3],10)+" "+(h===0?"00":"12")+"Z";\n'
'        return;\n'
'      }\n'
'    }\n'
'    for (var i=_SYN_TS_LIST.length-1; i>=0; i--) {\n'
'      var _m=_SYN_TS_LIST[i].match(/^(\\d{2})(\\d{2})(\\d{2})Z$/);\n'
'      if (!_m) continue;\n'
'      var _hh=parseInt(_m[2],10);\n'
'      var _matches=(h===0)?(_hh>=0&&_hh<6):(_hh>=12&&_hh<18);\n'
'      if (_matches){btn.textContent=_m[1]+"d "+(h===0?"00":h)+"Z";break;}\n'
'      if (i===0){btn.textContent=_m[1]+"d "+(h===0?"00":h)+"Z";break;}\n'
'    }\n'
'  });\n'
    '  if (latest) {\n'
    '    sel.value = latest;\n'
    '    var uaSel  = document.getElementById("ua-level-sel");\n'
    '    var sfcSel = document.getElementById("sfc-master-sel");\n'
    '    _synUAHourKey = (_targetHour >= 12 && _targetHour < 18) ? "12" : "0";\n'
    '    _synBtnOff("btn-f00"); _synBtnOff("btn-f12");\n'
    '    var _fb = (_targetHour>=12&&_targetHour<18)?"btn-f12":"btn-f00";\n'
    '    _synBtnOn(_fb);\n'
    '    if (uaSel)  { uaSel.value  = "850"; _synUALevel = "850"; }\n'
    '    if (sfcSel) { sfcSel.value = "off"; }\n'
    '    ["sfc","850","700","500","250","analysis"].forEach(function(m) {\n'
    '      var b = document.getElementById("btn-mode-"+m);\n'
    '      if (b) b.style.background = (m === "850") ? "#4a7fc1" : "#b0b8c8";\n'
    '    });\n'
    '    _synShowUAStns = true; _synBtnOn("btn-ua-stns");\n'
    '    synUpdateTS(latest);\n'
    '  }\n'
    '}\n'
    'if (document.readyState==="complete") { setTimeout(synInitDropdown,500); }\n'
    'else { window.addEventListener("load",function(){setTimeout(synInitDropdown,500);}); }\n'
    '</script>\n'
)

m.get_root().html.add_child(Element(ts_js))

# ---- BORDERS JS ----------------------------------------------------------
borders_js = (
    '<style>.leaflet-container { background: #ffffff !important; }</style>\n'
    '<script>\n'
    '(function() {\n'
    '  function loadBorders() {\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if (!keys.length) { setTimeout(loadBorders,200); return; }\n'
    '    var MAP = window[keys[0]];\n'
    '    var items = [\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",\n'
    '       {color:"#444",weight:1.8,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson",\n'
    '       {color:"#333",weight:2.0,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces_lines.geojson",\n'
    '       {color:"#777",weight:0.9,opacity:0.85,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",\n'
    '       {color:"#5588aa",weight:0.8,opacity:0.9,fill:false}]\n'
    '    ];\n'
    '    items.forEach(function(item) {\n'
    '      fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '        L.geoJSON(gj,{style:function(){return item[1];}}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("border load failed",e);});\n'
    '    });\n'
    '    fetch("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson")\n'
    '      .then(function(r){return r.json();})\n'
    '      .then(function(gj){\n'
    '        var ab={type:"FeatureCollection",features:gj.features.filter(function(f){return f.properties.name==="Alberta";})};\n'
    '        L.geoJSON(ab,{style:function(){return {color:"#cc0000",weight:2.5,opacity:1.0,fill:false};}}).addTo(MAP);\n'
    '      }).catch(function(e){console.warn("Alberta border load failed",e);});\n'
    '  }\n'
    '  if (document.readyState==="complete") { setTimeout(loadBorders,600); }\n'
    '  else { window.addEventListener("load",function(){setTimeout(loadBorders,600);}); }\n'
    '})();\n'
    '</script>'
)
m.get_root().html.add_child(Element(borders_js))

# ---- SAVE PNG / HTML BUTTONS (top right) ---------------------------------
EXPORT_TIME = '1200Z'   # ← change to whatever default export time you want

save_btn_html = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>\n'
    '<div id="syn-save-bar" style="'
    'position:fixed;bottom:78px;right:10px;z-index:10000;'
    'background:rgba(255,255,255,0.96);border:1px solid #ccc;border-radius:8px;'
    'padding:6px 12px;font-family:Courier New,monospace;font-size:12px;'
    'box-shadow:0 2px 10px rgba(0,0,0,0.15);display:flex;align-items:center;gap:8px;">'
    '<button id="btn-save-png" onclick="synSavePNG()" '
'style="font-size:11px;padding:4px 12px;cursor:pointer;border:1px solid #aaa;'
'border-radius:4px;background:#4a7fc1;color:#fff;font-family:Courier New,monospace;">'
'Save 850mb</button>'
    '<button id="btn-save-500mb" onclick="synSave500mb()" '
    'style="font-size:11px;padding:4px 12px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:4px;background:#4a7fc1;color:#fff;font-family:Courier New,monospace;">'
    'Save 500mb</button>'
    '<span id="save-status" style="color:#888;font-size:10px;"></span>'
    '</div>'
    '<script>\n'

    'function synSavePNG() {\n'
    '  var btn    = document.getElementById("btn-save-png");\n'
    '  var status = document.getElementById("save-status");\n'
    '  btn.disabled = true;\n'
    '  btn.textContent = "Capturing...";\n'
    '  status.textContent = "";\n'
    '  var keys = Object.keys(window).filter(function(k){ return k.startsWith("map_"); });\n'
    '  if (!keys.length) { status.textContent="Map not found"; btn.disabled=false; return; }\n'
    '  var MAP   = window[keys[0]];\n'
    '  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");\n'
    '  if (!mapEl) { status.textContent="Map el not found"; btn.disabled=false; return; }\n'
'\n'
'  synSetMode(\'850\');\n'
'\n'

    # hide UI chrome
    '  var hideEls = [\n'
    '    mapEl.querySelector(".leaflet-control-container"),\n'
    '    document.querySelector(".leaflet-control-layers"),\n'
    '    document.querySelector(".leaflet-control-zoom"),\n'
    '    document.querySelector(".leaflet-control-attribution"),\n'
    '    document.getElementById("syn-ts-bar"),\n'
    '    document.getElementById("syn-save-bar"),\n'
    '    document.getElementById("syn-mode-bar"),\n'
    '    document.getElementById("syn-fs-btn"),\n'
    '    document.getElementById("syn-ts-display")\n'
    '  ].filter(Boolean);\n'
    '  var prevVis = hideEls.map(function(el){ return el.style.visibility; });\n'
    '  hideEls.forEach(function(el){ el.style.visibility = "hidden"; });\n'

    # resize map for clean capture
    '  var CENTER   = [55, -104];\n'
    '  var ZOOM     = 5;\n'
    '  var TARGET_W = 1400;\n'
    '  var TARGET_H = 1100;\n'
    # Map center and Zoom for print
    '  var origW = mapEl.style.width;\n'
    '  var origH = mapEl.style.height;\n'
    '  function restore() {\n'
    '    mapEl.style.width  = origW;\n'
    '    mapEl.style.height = origH;\n'
    '    MAP.invalidateSize();\n'
    '    btn.disabled = false;\n'
    '    btn.textContent = "Save 850mb";\n'

    '    hideEls.forEach(function(el,i){ el.style.visibility = prevVis[i]; });\n'
    '  }\n'
    '  mapEl.style.width  = TARGET_W + "px";\n'
    '  mapEl.style.height = TARGET_H + "px";\n'
    '  MAP.invalidateSize();\n'
    '  MAP.setView(CENTER, ZOOM, { animate: false });\n'
    '\n'
    '  setTimeout(function() {\n'
    '    html2canvas(mapEl, {\n'
    '      useCORS: true, allowTaint: true,\n'
    '      scale: 2, logging: false,\n'
    '      width: TARGET_W, height: TARGET_H\n'
    '    }).then(function(canvas) {\n'

    # crop to 8.5x11 proportions
    '      var cropH = canvas.height;\n'
    '      var cropW = Math.min(Math.round(cropH * 8.5 / 11.0), canvas.width);\n'
    '      var out = document.createElement("canvas");\n'
    '      out.width  = cropW;\n'
    '      out.height = cropH;\n'
    '      var ctx2 = out.getContext("2d");\n'
    '      ctx2.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);\n'

    # white margin strips
    '      var MARGIN = 36;\n'
    '      ctx2.fillStyle = "rgba(255,255,255,1.0)";\n'
    '      ctx2.fillRect(0,              0,              cropW,  MARGIN);\n'
    '      ctx2.fillRect(0,              cropH - MARGIN, cropW,  MARGIN);\n'
    '      ctx2.fillRect(0,              0,              MARGIN, cropH);\n'
    '      ctx2.fillRect(cropW - MARGIN, 0,              MARGIN, cropH);\n'

    # timestamp label box bottom-left
    '      var today  = new Date();\n'
    '      var months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];\n'
    '      var dows   = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];\n'
    '      var dowStr  = dows[today.getUTCDay()];\n'
    '      var dateStr = months[today.getUTCMonth()] + " " + String(today.getUTCDate()).padStart(2,"0") + " " + today.getUTCFullYear();\n'
    '      var selEl   = document.getElementById("ts-select");\n'
    '      var tsVal   = selEl ? selEl.value : "";\n'
    '      var timeStr = tsVal ? tsVal.slice(2) : "1200Z";\n'
    '      var uaLvlEl = document.getElementById("ua-level-sel");\n'
    '      var uaLvl   = uaLvlEl ? uaLvlEl.value : "";\n'
    '      var mapTitle = "850MB ANALYSIS MAP";\n'
    '      var lines   = [mapTitle, dowStr + " " + dateStr, timeStr];\n'
    '      var fSize   = 36;\n'
    '      var pad     = 24;\n'
    '      var lineH   = fSize * 1.3;\n'
    '      var boxH    = lines.length * lineH + pad * 2;\n'
    '      ctx2.font   = fSize + "px Arial, sans-serif";\n'
    '      var maxW    = Math.max.apply(null, lines.map(function(l){ return ctx2.measureText(l).width; }));\n'
    '      var boxW    = maxW + pad * 2;\n'
    '      var bx      = MARGIN;\n'
    '      var by      = cropH - MARGIN - boxH;\n'
    '      ctx2.fillStyle = "rgba(255,255,255,0.88)";\n'
    '      ctx2.fillRect(bx, by, boxW, boxH);\n'
    '      ctx2.strokeStyle = "#1a4a8a";\n'
    '      ctx2.lineWidth = 3;\n'
    '      ctx2.strokeRect(bx, by, boxW, boxH);\n'
    '      ctx2.fillStyle    = "#1a2030";\n'
    '      ctx2.textBaseline = "top";\n'
    '      ctx2.textAlign    = "center";\n'
    '      var centerX = bx + boxW / 2;\n'
    '      lines.forEach(function(line, i) {\n'
    '        ctx2.font = fSize + "px Arial, sans-serif";\n'
    '        ctx2.fillText(line, centerX, by + pad + i * lineH);\n'
    '      });\n'

    # frame border
    '      ctx2.strokeStyle = "#1a2030";\n'
    '      ctx2.lineWidth   = 2;\n'
    '      ctx2.strokeRect(MARGIN, MARGIN, cropW - MARGIN * 2, cropH - MARGIN * 2);\n'
    '      var MAP2 = window[Object.keys(window).filter(function(k){return k.startsWith("map_");})[0]];\n'
    '      if (MAP2) {\n'
    '        var _corners = [\n'
    '          { px: MARGIN+4,        py: MARGIN+4,        ax:"left",   ay:"top",    pt: MAP2.containerPointToLatLng([0,0]) },\n'
    '          { px: cropW-MARGIN-4,  py: MARGIN+4,        ax:"right",  ay:"top",    pt: MAP2.containerPointToLatLng([TARGET_W,0]) },\n'
    '          { px: MARGIN+4,        py: cropH-MARGIN-4,  ax:"left",   ay:"bottom", pt: MAP2.containerPointToLatLng([0,TARGET_H]) },\n'
    '          { px: cropW-MARGIN-4,  py: cropH-MARGIN-4,  ax:"right",  ay:"bottom", pt: MAP2.containerPointToLatLng([TARGET_W,TARGET_H]) },\n'
    '        ];\n'
    '        ctx2.font = "20px Courier New, monospace";\n'
    '        ctx2.fillStyle = "#1a2030";\n'
    '        ctx2.textBaseline = "top";\n'
    '        _corners.forEach(function(c) {\n'
    '          var lat = c.pt.lat.toFixed(1);\n'
    '          var lon = c.pt.lng.toFixed(1);\n'
    '          var line1 = (lat >= 0 ? lat+"°N" : Math.abs(lat)+"°S");\n'
    '          var line2 = (lon >= 0 ? lon+"°E" : Math.abs(lon)+"°W");\n'
    '          ctx2.textAlign    = c.ax;\n'
    '          ctx2.textBaseline = c.ay === "top" ? "bottom" : "top";\n'
    '          var lineH2 = 22;\n'
    '          var outerY = c.ay === "top" ? MARGIN - 4 : cropH - MARGIN + 4;\n'
    '          ctx2.fillText(line1, c.px, c.ay === "top" ? outerY - lineH2 : outerY);\n'
    '          ctx2.fillText(line2, c.px, c.ay === "top" ? outerY          : outerY + lineH2);\n'
    '        });\n'
    '      }\n'
    # download
    '      var ts   = (document.getElementById("ts-select") || {}).value || "synoptic";\n'
    '      var dows2  = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];\n'
'      var months2= ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];\n'
'      var now2   = new Date();\n'
'      var dowStr2  = dows2[now2.getUTCDay()];\n'
'      var monStr2  = months2[now2.getUTCMonth()];\n'
'      var dayStr2  = String(now2.getUTCDate()).padStart(2,"0");\n'
'      var yrStr2   = now2.getUTCFullYear();\n'
'      var selEl2   = document.getElementById("ts-select");\n'
'      var tsVal2   = selEl2 ? selEl2.value : "";\n'
'      var timeStr2 = tsVal2 ? tsVal2.slice(2) : "1200Z";\n'
'      var uaLvlEl2 = document.getElementById("ua-level-sel");\n'
'      var uaLvl2   = uaLvlEl2 ? uaLvlEl2.value : "";\n'
'      var titleStr2 = "850MB_ANALYSIS_MAP";\n'
'      var name = (titleStr2 + "_" + dowStr2 + "_" + monStr2 + "_" + dayStr2 + "_" + yrStr2 + "_" + timeStr2 + ".png").replace(/[^a-zA-Z0-9_\\.]/g, "_");\n'
    '      var link = document.createElement("a");\n'
    '      link.download = name;\n'
    '      link.href = out.toDataURL("image/png");\n'
    '      link.click();\n'
    '      restore();\n'
    '      status.textContent = "Saved!";\n'
    '      setTimeout(function(){ status.textContent = ""; }, 3000);\n'
    '    }).catch(function(e) {\n'
    '      restore();\n'
    '      status.textContent = "Failed: " + e.message;\n'
    '    });\n'
    '  }, 600);\n'
    '}\n'



'function synSave500mb() {\n'
'  var btn    = document.getElementById("btn-save-500mb");\n'
'  var status = document.getElementById("save-status");\n'
'  btn.disabled = true;\n'
'  btn.textContent = "Capturing...";\n'
'  status.textContent = "";\n'
'  var keys = Object.keys(window).filter(function(k){ return k.startsWith("map_"); });\n'
'  if (!keys.length) { status.textContent="Map not found"; btn.disabled=false; return; }\n'
'  var MAP   = window[keys[0]];\n'
'  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");\n'
'  if (!mapEl) { status.textContent="Map el not found"; btn.disabled=false; return; }\n'

# Switch to 500mb mode first
'  synSetMode("500");\n'

# hide UI chrome
'  var hideEls = [\n'
'    mapEl.querySelector(".leaflet-control-container"),\n'
'    document.querySelector(".leaflet-control-layers"),\n'
'    document.querySelector(".leaflet-control-zoom"),\n'
'    document.querySelector(".leaflet-control-attribution"),\n'
'    document.getElementById("syn-ts-bar"),\n'
'    document.getElementById("syn-save-bar"),\n'
'    document.getElementById("syn-mode-bar"),\n'
'    document.getElementById("syn-fs-btn"),\n'
'    document.getElementById("syn-ts-display")\n'
'  ].filter(Boolean);\n'
'  var prevVis = hideEls.map(function(el){ return el.style.visibility; });\n'
'  hideEls.forEach(function(el){ el.style.visibility = "hidden"; });\n'

# resize map for clean capture
'  var CENTER   = [55, -118];\n'
'  var ZOOM     = 5;\n'
'  var TARGET_W = 1400;\n'
'  var TARGET_H = 1141;\n'
'  var origW = mapEl.style.width;\n'
'  var origH = mapEl.style.height;\n'
'  function restore() {\n'
'    mapEl.style.width  = origW;\n'
'    mapEl.style.height = origH;\n'
'    MAP.invalidateSize();\n'
'    btn.disabled = false;\n'
'    btn.textContent = "Save 500mb";\n'
'    hideEls.forEach(function(el,i){ el.style.visibility = prevVis[i]; });\n'
'  }\n'
'  mapEl.style.width  = TARGET_W + "px";\n'
'  mapEl.style.height = TARGET_H + "px";\n'
'  MAP.invalidateSize();\n'
'  MAP.setView(CENTER, ZOOM, { animate: false });\n'
'\n'
'  setTimeout(function() {\n'
'    html2canvas(mapEl, {\n'
'      useCORS: true, allowTaint: true,\n'
'      scale: 2, logging: false,\n'
'      width: TARGET_W, height: TARGET_H\n'
'    }).then(function(canvas) {\n'
'      var cropH = canvas.height;\n'
'      var cropW = Math.min(Math.round(cropH * 2944.0 / 2400.0), canvas.width);\n'
'      var out = document.createElement("canvas");\n'
'      out.width  = cropW;\n'
'      out.height = cropH;\n'
'      var ctx2 = out.getContext("2d");\n'
'      ctx2.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);\n'
'      var MARGIN = 36;\n'
'      ctx2.fillStyle = "rgba(255,255,255,1.0)";\n'
'      ctx2.fillRect(0,              0,              cropW,  MARGIN);\n'
'      ctx2.fillRect(0,              cropH - MARGIN, cropW,  MARGIN);\n'
'      ctx2.fillRect(0,              0,              MARGIN, cropH);\n'
'      ctx2.fillRect(cropW - MARGIN, 0,              MARGIN, cropH);\n'
'      var today  = new Date();\n'
'      var months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];\n'
'      var dows   = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];\n'
'      var dowStr  = dows[today.getUTCDay()];\n'
'      var dateStr = months[today.getUTCMonth()] + " " + String(today.getUTCDate()).padStart(2,"0") + " " + today.getUTCFullYear();\n'
'      var selEl   = document.getElementById("ts-select");\n'
'      var tsVal   = selEl ? selEl.value : "";\n'
'      var timeStr = tsVal ? tsVal.slice(2) : "1200Z";\n'
'      var lines   = ["500MB ANALYSIS MAP", dowStr + " " + dateStr, timeStr];\n'
'      var fSize   = 36;\n'
'      var pad     = 24;\n'
'      var lineH   = fSize * 1.3;\n'
'      var boxH    = lines.length * lineH + pad * 2;\n'
'      ctx2.font   = fSize + "px Arial, sans-serif";\n'
'      var maxW    = Math.max.apply(null, lines.map(function(l){ return ctx2.measureText(l).width; }));\n'
'      var boxW    = maxW + pad * 2;\n'
'      var bx      = MARGIN;\n'
'      var by      = cropH - MARGIN - boxH;\n'
'      ctx2.fillStyle = "rgba(255,255,255,0.88)";\n'
'      ctx2.fillRect(bx, by, boxW, boxH);\n'
'      ctx2.strokeStyle = "#1a4a8a";\n'
'      ctx2.lineWidth = 3;\n'
'      ctx2.strokeRect(bx, by, boxW, boxH);\n'
'      ctx2.fillStyle    = "#1a2030";\n'
'      ctx2.textBaseline = "top";\n'
'      ctx2.textAlign    = "center";\n'
'      var centerX = bx + boxW / 2;\n'
'      lines.forEach(function(line, i) {\n'
'        ctx2.font = fSize + "px Arial, sans-serif";\n'
'        ctx2.fillText(line, centerX, by + pad + i * lineH);\n'
'      });\n'
'      ctx2.strokeStyle = "#1a2030";\n'
'      ctx2.lineWidth   = 2;\n'
'      ctx2.strokeRect(MARGIN, MARGIN, cropW - MARGIN * 2, cropH - MARGIN * 2);\n'
'      var MAP2 = window[Object.keys(window).filter(function(k){return k.startsWith("map_");})[0]];\n'
'      if (MAP2) {\n'
'        var _corners = [\n'
'          { px: MARGIN+4,        py: MARGIN+4,        ax:"left",   ay:"top",    pt: MAP2.containerPointToLatLng([0,0]) },\n'
'          { px: cropW-MARGIN-4,  py: MARGIN+4,        ax:"right",  ay:"top",    pt: MAP2.containerPointToLatLng([TARGET_W,0]) },\n'
'          { px: MARGIN+4,        py: cropH-MARGIN-4,  ax:"left",   ay:"bottom", pt: MAP2.containerPointToLatLng([0,TARGET_H]) },\n'
'          { px: cropW-MARGIN-4,  py: cropH-MARGIN-4,  ax:"right",  ay:"bottom", pt: MAP2.containerPointToLatLng([TARGET_W,TARGET_H]) },\n'
'        ];\n'
'        ctx2.font = "20px Courier New, monospace";\n'
'        ctx2.fillStyle = "#1a2030";\n'
'        ctx2.textBaseline = "top";\n'
'        _corners.forEach(function(c) {\n'
'          var lat = c.pt.lat.toFixed(1);\n'
'          var lon = c.pt.lng.toFixed(1);\n'
'          var line1 = (lat >= 0 ? lat+"°N" : Math.abs(lat)+"°S");\n'
'          var line2 = (lon >= 0 ? lon+"°E" : Math.abs(lon)+"°W");\n'
'          ctx2.textAlign    = c.ax;\n'
'          ctx2.textBaseline = c.ay === "top" ? "bottom" : "top";\n'
'          var lineH2 = 22;\n'
'          var outerY = c.ay === "top" ? MARGIN - 4 : cropH - MARGIN + 4;\n'
'          ctx2.fillText(line1, c.px, c.ay === "top" ? outerY - lineH2 : outerY);\n'
'          ctx2.fillText(line2, c.px, c.ay === "top" ? outerY          : outerY + lineH2);\n'
'        });\n'
'      }\n'
'      var dows2  = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];\n'
'      var months2= ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];\n'
'      var now2   = new Date();\n'
'      var dowStr2  = dows2[now2.getUTCDay()];\n'
'      var monStr2  = months2[now2.getUTCMonth()];\n'
'      var dayStr2  = String(now2.getUTCDate()).padStart(2,"0");\n'
'      var yrStr2   = now2.getUTCFullYear();\n'
'      var selEl2   = document.getElementById("ts-select");\n'
'      var tsVal2   = selEl2 ? selEl2.value : "";\n'
'      var timeStr2 = tsVal2 ? tsVal2.slice(2) : "1200Z";\n'
'      var name = ("500MB_ANALYSIS_MAP_" + dowStr2 + "_" + monStr2 + "_" + dayStr2 + "_" + yrStr2 + "_" + timeStr2 + ".png").replace(/[^a-zA-Z0-9_\\.]/g, "_");\n'
'      var link = document.createElement("a");\n'
'      link.download = name;\n'
'      link.href = out.toDataURL("image/png");\n'
'      link.click();\n'
'      restore();\n'
'      status.textContent = "Saved!";\n'
'      setTimeout(function(){ status.textContent = ""; }, 3000);\n'
'    }).catch(function(e) {\n'
'      restore();\n'
'      status.textContent = "Failed: " + e.message;\n'
'    });\n'
'  }, 600);\n'
'}\n'



    '</script>\n'
)
m.get_root().html.add_child(Element(save_btn_html))

# ---- GITHUB ACTIONS TRIGGER BUTTON ---------------------------------------
GH_TRIGGER_PREFIX = 'ghp_5te1jZS2kbyfzeYUANY6CebGtQGpza2j'
GH_REPO           = 'ngsmetadvisor/UAanalysis'
GH_WORKFLOW       = 'upper_air_analysis.yml'
GH_BRANCH         = 'main'

gh_trigger_html = (
    '<style>\n'
    '#syn-gh-bar {\n'
    '  position:fixed;bottom:115px;right:10px;z-index:10002;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #ccc;border-radius:8px;\n'
    '  padding:6px 12px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 10px rgba(0,0,0,0.15);display:flex;align-items:center;gap:8px;\n'
    '}\n'
    '#gh-trigger-panel {\n'
    '  position:fixed;bottom:115px;right:10px;z-index:10002;\n'
    '  background:rgba(255,255,255,0.97);border:1px solid #ccc;border-radius:8px;\n'
    '  padding:10px 14px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 10px rgba(0,0,0,0.15);display:none;min-width:290px;\n'
    '}\n'
    '#gh-trigger-panel .gh-row { display:flex;align-items:center;gap:6px;margin-top:8px; }\n'
    '#gh-suffix-inp {\n'
    '  width:52px;font-family:Courier New,monospace;font-size:13px;\n'
    '  border:1px solid #aaa;border-radius:4px;padding:3px 6px;\n'
    '  text-align:center;letter-spacing:2px;\n'
    '}\n'
    '#gh-run-btn {\n'
    '  font-size:11px;padding:4px 10px;cursor:pointer;\n'
    '  border:1px solid #aaa;border-radius:4px;\n'
    '  background:#2a6a2a;color:#fff;font-family:Courier New,monospace;\n'
    '}\n'
    '#gh-run-btn:disabled { opacity:0.5;cursor:not-allowed; }\n'
    '#gh-status-msg { font-size:11px;margin-top:6px;padding:4px 8px;border-radius:4px;display:none; }\n'
    '#gh-status-msg.ok   { background:#e6faf0;color:#145c2c;border:1px solid #1a7a3a; }\n'
    '#gh-status-msg.fail { background:#fdecea;color:#a02020;border:1px solid #cc3333; }\n'
    '.gh-cancel-btn { font-size:11px;padding:4px 8px;cursor:pointer;border:1px solid #aaa;border-radius:4px;background:none;font-family:Courier New,monospace; }\n'
    '.gh-eye-btn { cursor:pointer;font-size:13px;background:none;border:none;padding:2px 4px;color:#555; }\n'
    '</style>\n'

    '<div id="syn-gh-bar" style="flex-direction:column;align-items:flex-start;gap:4px;">'
    '<button onclick="ghShowPanel()" '
    'style="font-size:11px;padding:4px 12px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:4px;background:#2a6a2a;color:#fff;font-family:Courier New,monospace;">'
    '&#9654; Run UA Analysis</button>'
    '<div id="gh-progress-wrap" style="display:none;width:180px;">'
    '  <div style="background:#ddd;border-radius:4px;height:6px;overflow:hidden;">'
    '    <div id="gh-progress-bar" style="height:6px;width:0%;background:#2a6a2a;'
    '    border-radius:4px;transition:width 0.4s ease;"></div>'
    '  </div>'
    '  <div id="gh-progress-label" style="font-size:9px;color:#555;margin-top:2px;'
    '  font-family:Courier New,monospace;">Queued…</div>'
    '</div>'
    '<div id="gh-last-run" style="font-size:9px;color:#888;font-family:Courier New,monospace;'
    'min-height:12px;"></div>'
    '</div>\n'

    '<div id="gh-trigger-panel">'
    '<b style="font-size:12px;color:#1a3a6a">Trigger upper_air_analysis.yml</b><br>'
    '<span style="font-size:10px;color:#888">ngsmetadvisor / UAanalysis &middot; main</span>'
    '<div class="gh-row">'
    '<span style="font-size:11px;color:#555;font-family:Courier New,monospace;letter-spacing:0.5px">'
    'PIN'
    '</span>'
    '<input id="gh-suffix-inp" type="password" maxlength="4" placeholder="????" '
    'autocomplete="off" spellcheck="false" oninput="ghClearStatus()" />'
    '<button class="gh-eye-btn" onclick="ghToggleEye()" title="show/hide">&#128065;</button>'
    '</div>'
    '<div class="gh-row">'
    '<button id="gh-run-btn" onclick="ghDispatch()">&#9654; Run</button>'
    '<button class="gh-cancel-btn" onclick="ghHidePanel()">Cancel</button>'
    '<span id="gh-spin" style="display:none;font-size:11px;color:#555">dispatching&hellip;</span>'
    '</div>'
    '<div id="gh-status-msg"></div>'
    '<a href="https://github.com/ngsmetadvisor/UAanalysis/actions" target="_blank" '
    'id="gh-actions-link" '
    'style="display:none;font-size:10px;color:#1a4a8a;margin-top:6px;text-decoration:none;">'
    '&#8599; View Actions on GitHub</a>'
    '<br><a href="https://ngsmetadvisor.github.io/UAanalysis/" target="_blank" '
    'style="font-size:10px;color:#1a4a8a;text-decoration:none;">'
    '&#8599; Open analysis page</a>'
    '</div>\n'

    '<script>\n'
    'var _GH_PREFIX   = "' + GH_TRIGGER_PREFIX + '";\n'
    'var _GH_REPO     = "' + GH_REPO + '";\n'
    'var _GH_WORKFLOW = "' + GH_WORKFLOW + '";\n'
    'var _GH_BRANCH   = "' + GH_BRANCH + '";\n'
    '\n'
    'function ghShowPanel() {\n'
    '  document.getElementById("gh-trigger-panel").style.display = "block";\n'
    '  document.getElementById("syn-gh-bar").style.display = "none";\n'
    '  setTimeout(function(){ document.getElementById("gh-suffix-inp").focus(); }, 50);\n'
    '}\n'
    'function ghHidePanel() {\n'
    '  document.getElementById("gh-trigger-panel").style.display = "none";\n'
    '  document.getElementById("syn-gh-bar").style.display = "flex";\n'
    '  ghClearStatus();\n'
    '}\n'
    'function ghToggleEye() {\n'
    '  var inp = document.getElementById("gh-suffix-inp");\n'
    '  inp.type = (inp.type === "password") ? "text" : "password";\n'
    '}\n'
    'function ghClearStatus() {\n'
    '  var s = document.getElementById("gh-status-msg");\n'
    '  s.style.display = "none"; s.className = ""; s.textContent = "";\n'
    '  document.getElementById("gh-run-btn").disabled = false;\n'
    '  document.getElementById("gh-spin").style.display = "none";\n'
    '  document.getElementById("gh-actions-link").style.display = "none";\n'
    '}\n'
    '\n'
    'var _ghPollTimer = null;\n'
    'var _ghPollStart = null;\n'
    'var _ghPollToken = null;\n'
    'var _GH_POLL_MAX_MS = 30 * 60 * 1000;\n'   # 30-minute timeout
    '\n'
    'function ghStartProgress(token) {\n'
    '  _ghPollToken = token;\n'
    '  _ghPollStart = Date.now();\n'
    '  document.getElementById("gh-progress-wrap").style.display = "block";\n'
    '  document.getElementById("gh-progress-bar").style.width = "5%";\n'
    '  document.getElementById("gh-progress-label").textContent = "Queued…";\n'
    '  _ghPollTimer = setInterval(ghPollRun, 12000);\n'
    '  setTimeout(ghPollRun, 5000);\n'
    '}\n'
    '\n'
    'function ghStopProgress(msg, success) {\n'
    '  clearInterval(_ghPollTimer); _ghPollTimer = null;\n'
    '  var bar = document.getElementById("gh-progress-bar");\n'
    '  bar.style.width = success ? "100%" : "100%";\n'
    '  bar.style.background = success ? "#2a6a2a" : "#cc2222";\n'
    '  document.getElementById("gh-progress-label").textContent = msg;\n'
    '  if (success) {\n'
    '    var now = new Date();\n'
    '    var pad = function(n){ return String(n).padStart(2,"0"); };\n'
    '    var lbl = now.getUTCFullYear()+"-"+pad(now.getUTCMonth()+1)+"-"+pad(now.getUTCDate())\n'
    '            + " "+pad(now.getUTCHours())+":"+pad(now.getUTCMinutes())+"Z";\n'
    '    document.getElementById("gh-last-run").textContent = "Last run: "+lbl;\n'
    '    localStorage.setItem("gh_last_run", lbl);\n'
    '  }\n'
    '  setTimeout(function(){\n'
    '    document.getElementById("gh-progress-wrap").style.display = "none";\n'
    '    document.getElementById("gh-progress-bar").style.background = "#2a6a2a";\n'
    '    document.getElementById("gh-progress-bar").style.width = "0%";\n'
    '  }, 4000);\n'
    '}\n'
    '\n'
    'function ghPollRun() {\n'
    '  if (!_ghPollToken) return;\n'
    '  if (Date.now() - _ghPollStart > _GH_POLL_MAX_MS) {\n'
    '    ghStopProgress("Timed out", false); return;\n'
    '  }\n'
    '  var elapsed = Math.min(95, Math.round((Date.now() - _ghPollStart) / _GH_POLL_MAX_MS * 100) + 5);\n'
    '  document.getElementById("gh-progress-bar").style.width = elapsed + "%";\n'
    '  fetch(\n'
    '    "https://api.github.com/repos/" + _GH_REPO + "/actions/runs?per_page=5&branch=" + _GH_BRANCH,\n'
    '    { headers: { "Authorization": "Bearer " + _ghPollToken,\n'
    '                 "Accept": "application/vnd.github+json",\n'
    '                 "X-GitHub-Api-Version": "2022-11-28" } }\n'
    '  ).then(function(r){ return r.json(); }).then(function(data) {\n'
    '    var runs = (data.workflow_runs || []).filter(function(r){\n'
    '      return r.path && r.path.indexOf(_GH_WORKFLOW) !== -1;\n'
    '    });\n'
    '    if (!runs.length) {\n'
    '      document.getElementById("gh-progress-label").textContent = "Waiting for run…";\n'
    '      return;\n'
    '    }\n'
    '    var run = runs[0];\n'
    '    var st  = run.status;\n'
    '    var con = run.conclusion;\n'
    '    var labels = { queued:"Queued…", in_progress:"Running…",\n'
    '                   completed:"Done", waiting:"Waiting…" };\n'
    '    document.getElementById("gh-progress-label").textContent = labels[st] || st;\n'
    '    if (st === "completed") {\n'
    '      var ok = (con === "success");\n'
    '      ghStopProgress(ok ? "✓ Success" : ("✗ " + (con||"failed")), ok);\n'
    '    }\n'
    '  }).catch(function(e) {\n'
    '    document.getElementById("gh-progress-label").textContent = "Poll error";\n'
    '  });\n'
    '}\n'
    '\n'
    '(function(){\n'
    '  function _restoreLastRun() {\n'
    '    var saved = localStorage.getItem("gh_last_run");\n'
    '    if (!saved) return;\n'
    '    var el = document.getElementById("gh-last-run");\n'
    '    if (el) { el.textContent = "Last run: " + saved; return; }\n'
    '    setTimeout(_restoreLastRun, 300);\n'
    '  }\n'
    '  if (document.readyState === "complete") { setTimeout(_restoreLastRun, 200); }\n'
    '  else { window.addEventListener("load", function(){ setTimeout(_restoreLastRun, 200); }); }\n'
    '})();\n'
    'function ghShowStatus(type, msg) {\n'
    '  var s = document.getElementById("gh-status-msg");\n'
    '  s.className = type; s.textContent = msg; s.style.display = "block";\n'
    '}\n'
    'function ghDispatch() {\n'
    '  var suffix = document.getElementById("gh-suffix-inp").value.trim();\n'
    '  if (suffix.length !== 4) {\n'
    '    ghShowStatus("fail", "Enter exactly 4 characters to complete the token.");\n'
    '    return;\n'
    '  }\n'
    '  var token = _GH_PREFIX + suffix;\n'
    '  var btn   = document.getElementById("gh-run-btn");\n'
    '  btn.disabled = true;\n'
    '  document.getElementById("gh-spin").style.display = "inline";\n'
    '  ghClearStatus();\n'
    '  fetch(\n'
    '    "https://api.github.com/repos/" + _GH_REPO +\n'
    '    "/actions/workflows/" + _GH_WORKFLOW + "/dispatches",\n'
    '    {\n'
    '      method: "POST",\n'
    '      headers: {\n'
    '        "Authorization": "Bearer " + token,\n'
    '        "Accept": "application/vnd.github+json",\n'
    '        "Content-Type": "application/json",\n'
    '        "X-GitHub-Api-Version": "2022-11-28"\n'
    '      },\n'
    '      body: JSON.stringify({ ref: _GH_BRANCH })\n'
    '    }\n'
    '  ).then(function(r) {\n'
    '    document.getElementById("gh-spin").style.display = "none";\n'
    '    if (r.status === 204) {\n'
    '      ghShowStatus("ok", "Workflow dispatched successfully!");\n'
    '      document.getElementById("gh-actions-link").style.display = "block";\n'
    '      ghHidePanel();\n'
    '      ghStartProgress(token);\n'
    '    } else {\n'
    '      return r.json().then(function(b) {\n'
    '        var msg = (b && b.message) ? b.message : ("HTTP " + r.status);\n'
    '        if (r.status === 401) msg += " — check last 4 characters.";\n'
    '        if (r.status === 422) msg += " — workflow may not have workflow_dispatch trigger.";\n'
    '        ghShowStatus("fail", "Error: " + msg);\n'
    '        btn.disabled = false;\n'
    '      });\n'
    '    }\n'
    '  }).catch(function(e) {\n'
    '    document.getElementById("gh-spin").style.display = "none";\n'
    '    ghShowStatus("fail", "Network error — " + e.message);\n'
    '    btn.disabled = false;\n'
    '  });\n'
    '}\n'
    'document.addEventListener("keydown", function(e) {\n'
    '  if (e.key === "Enter" && document.getElementById("gh-trigger-panel").style.display === "block") {\n'
    '    ghDispatch();\n'
    '  }\n'
    '  if (e.key === "Escape" && document.getElementById("gh-trigger-panel").style.display === "block") {\n'
    '    ghHidePanel();\n'
    '  }\n'
    '});\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(gh_trigger_html))

# ---- FULLSCREEN BUTTON ---------------------------------------------------
fullscreen_html = (
    '<style>\n'
    '#syn-fs-btn {\n'
    '  position:fixed;top:10px;left:10px;z-index:10001;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;\n'
    '  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;\n'
    '}\n'
    '#syn-fs-btn:hover { background:#e8f0fe; }\n'
    '</style>\n'
    '<button id="syn-fs-btn" onclick="synToggleFS()">&#x26F6; Fullscreen</button>\n'
    '<div id="syn-ts-display" style="'
    'position:fixed;top:10px;left:120px;z-index:10001;'
    'background:rgba(255,255,255,0.92);border:1px solid #aaa;border-radius:6px;'
    'padding:5px 10px;font-family:Courier New,monospace;font-size:12px;'
    'box-shadow:0 2px 8px rgba(0,0,0,0.15);color:#1a3a6a;pointer-events:none;'
    '"></div>\n'
    '<script>\n'
    'var _synFS=false, _synMapEl=null, _synOrigStyle="";\n'
    'function synToggleFS() {\n'
    '  var btn=document.getElementById("syn-fs-btn");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if (!_synMapEl) { _synMapEl=document.getElementById(keys[0])||document.querySelector(".leaflet-container"); }\n'
    '  if (!_synMapEl) return;\n'
    '  _synFS=!_synFS;\n'
    '  if (_synFS) {\n'
    '    _synOrigStyle=_synMapEl.getAttribute("style")||"";\n'
    '    _synMapEl.setAttribute("style","position:fixed!important;top:0;left:0;width:100vw!important;height:100vh!important;z-index:9999!important;margin:0!important;");\n'
    '    btn.innerHTML="&#x274C; Exit Fullscreen";\n'
    '  } else {\n'
    '    _synMapEl.setAttribute("style",_synOrigStyle);\n'
    '    btn.innerHTML="&#x26F6; Fullscreen";\n'
    '  }\n'
    '  setTimeout(function(){MAP.invalidateSize();},100);\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(fullscreen_html))
m.get_root().html.add_child(Element(tooltip_toggle_html))

mode_bar_html = (
    '<div id="syn-mode-bar" style="'
    'position:fixed;bottom:78px;left:10px;z-index:10001;'
    'background:rgba(255,255,255,0.96);border:1px solid #ccc;border-radius:8px;'
    'padding:5px 10px;font-family:Courier New,monospace;font-size:11px;'
    'box-shadow:0 2px 8px rgba(0,0,0,0.15);display:flex;align-items:center;gap:6px;">'
    '<b style="color:#555;font-size:9px">MODE</b>'
    '<button id="btn-mode-sfc" onclick="synSetMode(\'sfc\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#b0b8c8;color:#fff">SFC</button>'
    '<button id="btn-mode-850" onclick="synSetMode(\'850\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#4a7fc1;color:#fff">850</button>'
    '<button id="btn-mode-700" onclick="synSetMode(\'700\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#b0b8c8;color:#fff">700</button>'
    '<button id="btn-mode-500" onclick="synSetMode(\'500\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#b0b8c8;color:#fff">500</button>'
    '<button id="btn-mode-250" onclick="synSetMode(\'250\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#b0b8c8;color:#fff">250</button>'
    '<span style="width:1px;background:#ccc;align-self:stretch;margin:0 2px;"></span>'
    '<button id="btn-mode-analysis" onclick="synSetMode(\'analysis\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#b0b8c8;color:#fff">Analysis</button>'
    '<button id="btn-mode-drymicroburst" onclick="synSetMode(\'drymicroburst\')" '
    'style="font-size:10px;padding:2px 8px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#b0b8c8;color:#fff">Dry Microburst</button>'
    '</div>\n'
    '<script>\n'
    'var _syn850DryLayer    = null; var _synShow850Dry    = false; var _syn850DryLastKey    = null;\n'
    'var _syn700DryLayer    = null; var _synShow700Dry    = false; var _syn700DryLastKey    = null;\n'
    'var _syn500DryLayer    = null; var _synShow500Dry    = false; var _syn500DryLastKey    = null;\n'
    'var _syn700MoistLayer  = null; var _synShow700Moist  = false; var _syn700MoistLastKey  = null;\n'
    'var _syn500MoistLayer  = null; var _synShow500Moist  = false; var _syn500MoistLastKey  = null;\n'

    'function _synRenderUATtdFill(lvl, threshold, above, lineCol, fillCol, btnId, layerVar, showVar, lastKeyVar) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  var uaKey = _synUAHourKey;\n'
    '  if (uaKey === window[lastKeyVar] && window[layerVar]) {\n'
    '    if (window[showVar]) window[layerVar].addTo(MAP);\n'
    '    return;\n'
    '  }\n'
    '  window[lastKeyVar] = uaKey;\n'
    '  if (window[layerVar]) { MAP.removeLayer(window[layerVar]); window[layerVar] = null; }\n'
    '  var uaData = (_SYN_UA[uaKey] || {levels:{}}).levels[String(lvl)] || {};\n'
    '  var lg = L.layerGroup();\n'
    '  (uaData.ttdp || []).forEach(function(ct) {\n'
    '    var passes = above ? (ct.level > threshold) : (ct.level <= threshold);\n'
    '    if (!passes) return;\n'
    '    var ll = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    L.polygon(ll, {color:lineCol, weight:1.5, opacity:0.85, dashArray:"6 4",\n'
    '      fillColor:fillCol, fillOpacity:0.25}).bindTooltip(lvl+" hPa T-Td="+ct.level.toFixed(1)+"°C").addTo(lg);\n'
    '  });\n'
    '  window[layerVar] = lg;\n'
    '  if (window[showVar]) window[layerVar].addTo(MAP);\n'
    '}\n'

    'function synToggle850Dry() {\n'
    '  _synShow850Dry = !_synShow850Dry;\n'
    '  if (_synShow850Dry) _synBtnOn("btn-850dry"); else _synBtnOff("btn-850dry");\n'
    '  _synRenderUATtdFill(850,20,true,"#cc0000","#8B4513","btn-850dry","_syn850DryLayer","_synShow850Dry","_syn850DryLastKey");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if (!_synShow850Dry && _syn850DryLayer) MAP.removeLayer(_syn850DryLayer);\n'
    '}\n'

    'function synToggle700Dry() {\n'
    '  _synShow700Dry = !_synShow700Dry;\n'
    '  if (_synShow700Dry) _synBtnOn("btn-700dry"); else _synBtnOff("btn-700dry");\n'
    '  _synRenderUATtdFill(700,10,true,"#8B4513","#8B4513","btn-700dry","_syn700DryLayer","_synShow700Dry","_syn700DryLastKey");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if (!_synShow700Dry && _syn700DryLayer) MAP.removeLayer(_syn700DryLayer);\n'
    '}\n'

    'function synToggle500Dry() {\n'
    '  _synShow500Dry = !_synShow500Dry;\n'
    '  if (_synShow500Dry) _synBtnOn("btn-500dry"); else _synBtnOff("btn-500dry");\n'
    '  _synRenderUATtdFill(500,10,true,"#0044cc","#8B4513","btn-500dry","_syn500DryLayer","_synShow500Dry","_syn500DryLastKey");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if (!_synShow500Dry && _syn500DryLayer) MAP.removeLayer(_syn500DryLayer);\n'
    '}\n'

    'function synToggle700Moist() {\n'
    '  _synShow700Moist = !_synShow700Moist;\n'
    '  if (_synShow700Moist) _synBtnOn("btn-700moist"); else _synBtnOff("btn-700moist");\n'
    '  _synRenderUATtdFill(700,2,false,"#8B4513","#add8e6","btn-700moist","_syn700MoistLayer","_synShow700Moist","_syn700MoistLastKey");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if (!_synShow700Moist && _syn700MoistLayer) MAP.removeLayer(_syn700MoistLayer);\n'
    '}\n'

    'function synToggle500Moist() {\n'
    '  _synShow500Moist = !_synShow500Moist;\n'
    '  if (_synShow500Moist) _synBtnOn("btn-500moist"); else _synBtnOff("btn-500moist");\n'
    '  _synRenderUATtdFill(500,10,false,"#0044cc","#add8e6","btn-500moist","_syn500MoistLayer","_synShow500Moist","_syn500MoistLastKey");\n'
    '  var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP=window[keys[0]];\n'
    '  if (!_synShow500Moist && _syn500MoistLayer) MAP.removeLayer(_syn500MoistLayer);\n'
    '}\n'

    'function _synResetAll() {\n'
    '  var uaSel  = document.getElementById("ua-level-sel");\n'
    '  var sfcSel = document.getElementById("sfc-master-sel");\n'
    '  if (uaSel)  { uaSel.value  = ""; synUpdateUALevel(""); }\n'
    '  if (sfcSel) { sfcSel.value = "off"; synSfcMaster("off"); }\n'
    '  if (_synShowThermalRidge)  synToggleThermalRidge();\n'
    '  if (_synShowThermalTrough) synToggleThermalTrough();\n'
    '  if (_synShowRidge700)      synToggleRidge700();\n'
    '  if (_synShowTrough700)     synToggleTrough700();\n'
    '  if (_synShowRidge500)      synToggleRidge500();\n'
    '  if (_synShowTrough500)     synToggleTrough500();\n'
    '  if (_synShowTtd)           synToggleLayer("ttd");\n'
    '  if (_synShow850Moist)      synToggle850Moist();\n'
    '  if (_synShowInstab)        synToggleInstab();\n'
    '  if (_synShowUAStns)        synToggleUAStns();\n'
    '  if (_synShowVort)          synToggleVort();\n'
    '  if (_synConvSfcShow)       synToggleConvSfc();\n'
    '  if (_synConv850Show)       synToggleConv850();\n'
    '  if (_synShowSfcTrough)     synToggleSfcTrough();\n'
    '  if (_synShowTendRing)      synToggleTendRing();\n'
    '  if (_synShow850Dry)        synToggle850Dry();\n'
    '  if (_synShow700Dry)        synToggle700Dry();\n'
    '  if (_synShow500Dry)        synToggle500Dry();\n'
    '  if (_synShow700Moist)      synToggle700Moist();\n'
    '  if (_synShow500Moist)      synToggle500Moist();\n'
    '}\n'
    'function synSetMode(mode) {\n'
    '  ["sfc","850","700","500","250","analysis"].forEach(function(m) {\n'
    '    var b = document.getElementById("btn-mode-"+m);\n'
    '    if (b) b.style.background = (m === mode) ? "#4a7fc1" : "#b0b8c8";\n'
    '  });\n'
    '  var uaSel = document.getElementById("ua-level-sel");\n'
    '  var sfcSel = document.getElementById("sfc-master-sel");\n'
    '  if (mode === "sfc") {\n'
    '    _synResetAll();\n'
    '    if (sfcSel) { sfcSel.value = "on"; synSfcMaster("on"); }\n'
    '  } else if (mode === "analysis") {\n'
    '    _synResetAll();\n'
    '    // 850 Ridge ON\n'
    '    if (!_synShowThermalRidge)  synToggleThermalRidge();\n'
    '    // 850 Trough OFF\n'
    '    if (_synShowThermalTrough)  synToggleThermalTrough();\n'
    '    // 700 Ridge OFF\n'
    '    if (_synShowRidge700)       synToggleRidge700();\n'
    '    // 700 Trough ON\n'
    '    if (!_synShowTrough700)     synToggleTrough700();\n'
    '    // 500 Ridge OFF\n'
    '    if (_synShowRidge500)       synToggleRidge500();\n'
    '    // 500 Trough ON\n'
    '    if (!_synShowTrough500)     synToggleTrough500();\n'
    '    // SFC Moist ON\n'
    '    if (!_synShowTtd)           synToggleLayer("ttd");\n'
    '    // 850 Moist ON\n'
    '    if (!_synShow850Moist)      synToggle850Moist();\n'
    '    // Instab ON\n'
    '    if (!_synShowInstab)        synToggleInstab();\n'
    '    // Vort500 ON\n'
    '    if (!_synShowVort)          synToggleVort();\n'
    '    // SfcConv ON\n'
    '    if (_synConvSfcShow)       synToggleConvSfc();\n'
    '    // SFC Trough ON\n'
    '    if (!_synShowSfcTrough)     synToggleSfcTrough();\n'
    '    // SFC Conv ON\n'
    '    if (!_synConvSfcShow)       synToggleConvSfc();\n'
    '    // 850 Conv ON\n'
    '    if (!_synConv850Show)       synToggleConv850();\n'
    '    // SFC Trough ON\n'
    '    if (!_synShowSfcTrough)     synToggleSfcTrough();\n'
    '    // P-Tend ON\n'
    '    if (!_synShowTendRing)      synToggleTendRing();\n'
    '  } else if (mode === "drymicroburst") {\n'
    '    _synResetAll();\n'
    '    // 850 Dry ON\n'
    '    if (!_synShow850Dry)        synToggle850Dry();\n'
    '    // 700 Dry ON\n'
    '    if (!_synShow700Dry)        synToggle700Dry();\n'
    '    // 500 Moist ON\n'
    '    if (!_synShow500Moist)      synToggle500Moist();\n'
    '    // Instab ON\n'
    '    if (!_synShowInstab)        synToggleInstab();\n'
    '  } else {\n'
    '    _synResetAll();\n'
    '    if (uaSel) { uaSel.value = mode; synUpdateUALevel(mode); }\n'
    '    if (!_synShowUAStns) { _synShowUAStns = true; _synBtnOn("btn-ua-stns"); }\n'
    '    var sel = document.getElementById("ts-select");\n'
    '    if (sel && sel.value) synRenderUA(sel.value);\n'
    '  }\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(mode_bar_html))

m.get_root().html.add_child(Element(fire_zones_html))

# ── Vorticity layer ───────────────────────────────────────────────────────
from datetime import datetime, timedelta
import json as _json_vort
import requests
from xml.etree import ElementTree as ET

EC_WMS = "https://geo.weather.gc.ca/geomet"
VORT_LAYER_NAME = "RDPS_10km_AbsoluteVorticity_500mb"

try:
    cap_resp = requests.get(
        f"{EC_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&LAYER={VORT_LAYER_NAME}",
        timeout=30
    )
    root_ec = ET.fromstring(cap_resp.content)

    raw_times = []
    for _lyr in root_ec.iter('{http://www.opengis.net/wms}Layer'):
        _nel = _lyr.find('{http://www.opengis.net/wms}Name')
        if _nel is not None and _nel.text == VORT_LAYER_NAME:
            for _dim in _lyr.iter('{http://www.opengis.net/wms}Dimension'):
                if _dim.get('name') == 'time' and _dim.text:
                    raw_times = [t.strip() for t in _dim.text.strip().split(',')]
                    break

    def _expand_times(raw):
        out = []
        for entry in raw:
            if '/' in entry:
                parts = entry.split('/')
                s = datetime.fromisoformat(parts[0].replace('Z', '+00:00'))
                e = datetime.fromisoformat(parts[1].replace('Z', '+00:00'))
                step = timedelta(hours=int(parts[2].replace('PT','').replace('H','')))
                t = s
                while t <= e:
                    out.append(t.strftime('%Y-%m-%dT%H:%M:%SZ'))
                    t += step
            else:
                out.append(entry)
        return out

    _ec_times = _expand_times(raw_times)
    print(f"RDPS times: {_ec_times[0]} → {_ec_times[-1]} ({len(_ec_times)} steps)")

    vort_time_map = {}
    _vort_no_match = []
    for _k, _ds in _ua_date_map.items():
        _dt = datetime.strptime(str(_ds).strip(), '%Y-%m-%d %HZ').replace(
            tzinfo=__import__('datetime').timezone.utc)
        _best = min(_ec_times, key=lambda t: abs(
            datetime.fromisoformat(t.replace('Z', '+00:00')) - _dt))
        _best_dt = datetime.fromisoformat(_best.replace('Z', '+00:00'))
        _diff_h  = abs((_best_dt - _dt).total_seconds()) / 3600
        vort_time_map[_k] = _best
        _match = "✓" if _diff_h <= 6 else f"⚠ {_diff_h:.0f}h off"
        print(f"UA {_k}Z ({_ds}) → {_best} {_match}")
        if _diff_h > 6:
            _vort_no_match.append(f"UA {_k}Z: nearest vort is {_diff_h:.0f}h away ({_best})")


    _vort_time_map_str = _json_vort.dumps(vort_time_map)
    _vort_no_match_str = _json_vort.dumps(_vort_no_match)

    # ── Pre-fetch vort tiles as base64 and embed directly ──────────────────
    import base64, requests as _req
    from PIL import Image
    import io

    def _fetch_vort_tile_b64(time_str, bbox="-170,40,-50,75", width=900, height=600):
        url = (
            f"https://geo.weather.gc.ca/geomet"
            f"?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&LAYERS={VORT_LAYER_NAME}"
            f"&BBOX={bbox}&CRS=EPSG:4326"
            f"&WIDTH={width}&HEIGHT={height}"
            f"&FORMAT=image/png&TRANSPARENT=TRUE"
            f"&TIME={time_str}"
        )
        try:
            r = _req.get(url, timeout=30)
            r.raise_for_status()
            b64 = base64.b64encode(r.content).decode()
            print(f"  ✓ fetched {time_str} ({len(r.content)//1024}KB)")
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            print(f"  ✗ failed {time_str}: {e}")
            return None

    print("Pre-fetching vort tiles...")
    _vort_images = {}
    for _k, _t in vort_time_map.items():
        _img = _fetch_vort_tile_b64(_t)
        if _img:
            _vort_images[_k] = _img

    _vort_images_str = _json_vort.dumps(_vort_images)
    print(f"Vort images fetched: {list(_vort_images.keys())}")

    _vort_time_map_str = _json_vort.dumps(vort_time_map)
    _vort_no_match_str = _json_vort.dumps(_vort_no_match)

    # ── Pre-fetch vort images as base64 with yellow/red filter ─────────────
    import base64 as _b64
    import requests as _req2
    import numpy as _np
    import io as _io
    from PIL import Image as _Image

    _VORT_BBOX   = "-170,40,-50,75"   # CRS:84 lon_min,lat_min,lon_max,lat_max
    _VORT_W, _VORT_H = 1200, 800


    def _fetch_vort_b64(time_str):
        _vurl = (
            f"https://geo.weather.gc.ca/geomet"
            f"?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&LAYERS={VORT_LAYER_NAME}"
            f"&BBOX={_VORT_BBOX}&CRS=CRS:84"
            f"&WIDTH={_VORT_W}&HEIGHT={_VORT_H}"
            f"&FORMAT=image/png&TRANSPARENT=TRUE"
            f"&TIME={time_str}&STYLES="
        )
        try:
            _r = _req2.get(_vurl, timeout=30)
            _r.raise_for_status()
            if _r.content[:4] != b'\x89PNG':
                print(f"  ✗ {time_str}: not a PNG — {_r.text[:200]}")
                return None

            # ── yellow/red filter ──────────────────────────────────────────
            _img = _Image.open(_io.BytesIO(_r.content)).convert("RGBA")
            _arr = _np.array(_img, dtype=_np.float32)
            _R, _G, _B = _arr[:,:,0], _arr[:,:,1], _arr[:,:,2]
            _is_yellow = (_R > 180) & (_G > 150) & (_B < 80)
            _is_red    = (_R > 180) & (_G < 100) & (_B < 80)
            _keep      = _is_yellow | _is_red
            _out       = _arr.copy().astype(_np.uint8)
            _out[:,:,3] = _np.where(_keep, 220, 0)
            _filtered  = _Image.fromarray(_out, 'RGBA')
            _buf       = _io.BytesIO()
            _filtered.save(_buf, format='PNG')
            _buf.seek(0)
            _b = _b64.b64encode(_buf.read()).decode()
            print(f"  ✓ {time_str} ({len(_r.content)//1024}KB raw, "
                  f"{int(_keep.sum()):,} px kept)")
            return f"data:image/png;base64,{_b}"
        except Exception as _e:
            print(f"  ✗ {time_str}: {_e}")
            return None

    print("Pre-fetching Vort500 images...")
    _vort_images = {}
    for _k, _t in vort_time_map.items():
        _img = _fetch_vort_b64(_t)
        if _img:
            _vort_images[_k] = _img
    print(f"Done: {list(_vort_images.keys())}")

    _vort_images_str = _json_vort.dumps(_vort_images)

    vort_js = (
        '<script>\n'
        'var _VORT_TIME_MAP   = ' + _vort_time_map_str + ';\n'
        'var _VORT_NO_MATCH   = ' + _vort_no_match_str + ';\n'
        'var _VORT_LAYER      = "' + VORT_LAYER_NAME + '";\n'
        'var _synVortLayer    = null;\n'
        'var _synShowVort     = false;\n'

        # ── show red warning banner ──
        'function _synVortWarn(msgs) {\n'
        '  var el = document.getElementById("syn-vort-warn");\n'
        '  if (!el) {\n'
        '    el = document.createElement("div");\n'
        '    el.id = "syn-vort-warn";\n'
        '    el.style.cssText = "position:fixed;top:90px;left:10px;z-index:10002;"\n'
        '      + "background:rgba(200,20,20,0.93);color:#fff;border-radius:6px;"\n'
        '      + "padding:5px 14px;font-family:Courier New,monospace;font-size:11px;"\n'
        '      + "box-shadow:0 2px 8px rgba(0,0,0,0.3);pointer-events:none;";\n'
        '    document.body.appendChild(el);\n'
        '  }\n'
        '  if (msgs && msgs.length) {\n'
        '    el.textContent = "⚠ Vort500: " + msgs.join(" | ");\n'
        '    el.style.display = "block";\n'
        '    setTimeout(function(){ el.style.display="none"; }, 6000);\n'
        '  } else {\n'
        '    el.style.display = "none";\n'
        '  }\n'
        '}\n'

        'function synToggleVort() {\n'
        '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
        '  if (!keys.length) return;\n'
        '  var MAP = window[keys[0]];\n'
        '  _synShowVort = !_synShowVort;\n'
        '  if (!_synShowVort) {\n'
        '    if (_synVortLayer) { MAP.removeLayer(_synVortLayer); _synVortLayer = null; }\n'
        '    _synBtnOff("btn-vort");\n'
        '    _synVortWarn([]);\n'
        '    return;\n'
        '  }\n'
        '  if (_VORT_NO_MATCH && _VORT_NO_MATCH.length) _synVortWarn(_VORT_NO_MATCH);\n'
        '  _synRenderVort(MAP);\n'
        '  _synBtnOn("btn-vort");\n'
        '}\n'

        'var _VORT_IMAGES = ' + _vort_images_str + ';\n'

        'function _synRenderVort(MAP) {\n'
        '  if (_synVortLayer) { MAP.removeLayer(_synVortLayer); _synVortLayer = null; }\n'
        '  var imgData = _VORT_IMAGES[_synUAHourKey] || _VORT_IMAGES["0"] || Object.values(_VORT_IMAGES)[0];\n'
        '  if (!imgData) { console.error("No vort image for key:", _synUAHourKey); return; }\n'
        '  var bounds = [[40, -170], [75, -50]];\n'
        '  _synVortLayer = L.imageOverlay(imgData, bounds, {\n'
        '    opacity: 0.75,\n'
        '    zIndex: 9000,\n'
        '    interactive: false\n'
        '  });\n'
        '  _synVortLayer.addTo(MAP);\n'
        '  console.log("Vort overlay added for key:", _synUAHourKey);\n'
        '}\n'


        # hook into synFilterHour so vort refreshes when UA time changes
        'var _origSynFilterHour = typeof synFilterHour !== "undefined" ? synFilterHour : null;\n'
        'synFilterHour = function(h) {\n'
        '  if (_origSynFilterHour) _origSynFilterHour(h);\n'
        '  if (_synShowVort) {\n'
        '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
        '    if (keys.length) _synRenderVort(window[keys[0]]);\n'
        '    if (_VORT_NO_MATCH && _VORT_NO_MATCH.length) _synVortWarn(_VORT_NO_MATCH);\n'
        '  }\n'
        '};\n'
        '</script>\n'
    )
    m.get_root().html.add_child(Element(vort_js))
    print("✓ Vorticity JS injected.")

except Exception as _vort_ex:
    print(f"⚠ Vorticity layer skipped: {_vort_ex}")


# ── Cell 9 ADD-ON: inject convergence zones into existing Folium map ───────
# Paste this block at the END of Cell 9, just before m.save() / display()
# Requires: _conv_json_str  (from Cell CONV)
# ──────────────────────────────────────────────────────────────────────────

from folium import Element



# Colors are baked as literals inside a JS triple-quote string — no interpolation bugs.
# SFC  → black (#1a1a1a) line, dark-grey (#555555) fill
# 850  → red   (#cc0000) line, faint-red (#ff6666) fill

_conv_js = "<script>\n" + \
    "var _CONV_DATA       = " + _conv_json_str + ";\n" + \
    r"""
var _synConvSfcLayer = null;
var _synConvSfcShow  = false;
var _synConv850Layer = null;
var _synConv850Show  = false;

function _drawConvSegs(segs, lg, isSfc) {
  var lineCol = isSfc ? "#1a1a1a" : "#cc0000";
  var fillCol = isSfc ? "#555555" : "#ff6666";
  var lbl     = isSfc ? "SFC"     : "850";
  (segs || []).forEach(function(seg) {
    if (!seg.coords || seg.coords.length < 3) return;
    var ll = seg.coords.map(function(c){ return [c[1], c[0]]; });

    L.polygon(ll, {
      color: "none", weight: 0,
      fillColor: fillCol, fillOpacity: 0.25
    }).addTo(lg);

    L.polyline(ll, {
      color: lineCol, weight: 2.0, opacity: 0.90,
      dashArray: "10 6"
    }).bindTooltip(lbl + " convergence").addTo(lg);

    var step = Math.max(1, Math.floor(ll.length / 8));
    for (var i = Math.floor(step / 2); i < ll.length; i += step) {
      var xHtml = '<div style="font-size:13px;font-weight:900;line-height:1;color:'
                + lineCol
                + ';font-family:Arial,sans-serif;text-shadow:1px 1px 0 #fff,'
                + '-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff">&#215;&#215;</div>';
      L.marker(ll[i], {
        icon: L.divIcon({ html: xHtml, iconSize:[18,18], iconAnchor:[6,8], className:"" }),
        interactive: false
      }).addTo(lg);
    }

    var lblHtml = '<div style="font-size:10px;font-weight:bold;white-space:nowrap;color:'
                + lineCol
                + ';font-family:Courier New,monospace;text-shadow:1px 1px 0 #fff,'
                + '-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff">'
                + lbl + ' CONV</div>';
    L.marker([seg.label_lat, seg.label_lon], {
      icon: L.divIcon({ html: lblHtml, iconSize:[70,14], iconAnchor:[35,7], className:"" })
    }).addTo(lg);
  });
}

function _getMap() {
  var k = Object.keys(window).filter(function(k){ return k.startsWith("map_"); });
  return k.length ? window[k[0]] : null;
}

var _synSfcTroughLayer = null;
var _synShowSfcTrough  = false;

function _drawTroughSegs(segs, lg) {
  var lineCol = "#8B4513";
  (segs || []).forEach(function(seg) {
    if (!seg.coords || seg.coords.length < 2) return;
    var ll = seg.coords.map(function(c){ return [c[1], c[0]]; });
        L.polyline(ll, {
      color: "#000000", weight: 3.5, opacity: 0.90,
      dashArray: "12 6"
    }).bindTooltip("SFC Trough").addTo(lg);
    L.marker([seg.label_lat, seg.label_lon], {
      icon: L.divIcon({
        html: '<div style="font-size:10px;font-weight:bold;white-space:nowrap;color:#000000;font-family:Courier New,monospace;text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff">SFC TRGH</div>',
        iconSize:[70,14], iconAnchor:[35,7], className:""
      })
    }).addTo(lg);
  });
}

function synToggleSfcTrough() {
  _synShowSfcTrough = !_synShowSfcTrough;
  var MAP = _getMap(); if (!MAP) return;
  if (!_synShowSfcTrough) {
    if (_synSfcTroughLayer) MAP.removeLayer(_synSfcTroughLayer);
    _synBtnOff("btn-sfc-trough"); return;
  }
  var sel = document.getElementById("ts-select");
  var ts  = sel ? sel.value : "";
  if (_synSfcTroughLayer) MAP.removeLayer(_synSfcTroughLayer);
  _synSfcTroughLayer = L.layerGroup();
  _drawTroughSegs((_CONV_DATA.sfc_trough || {})[ts] || [], _synSfcTroughLayer);
  _synSfcTroughLayer.addTo(MAP);
  _synBtnOn("btn-sfc-trough");
}

function synToggleConvSfc() {
  _synConvSfcShow = !_synConvSfcShow;
  var MAP = _getMap(); if (!MAP) return;
  if (!_synConvSfcShow) {
    if (_synConvSfcLayer) MAP.removeLayer(_synConvSfcLayer);
    _synBtnOff("btn-conv-sfc"); return;
  }
  var sel = document.getElementById("ts-select");
  var ts  = sel ? sel.value : "";
  if (_synConvSfcLayer) MAP.removeLayer(_synConvSfcLayer);
  _synConvSfcLayer = L.layerGroup();
  _drawConvSegs((_CONV_DATA.sfc || {})[ts] || [], _synConvSfcLayer, true);
  _synConvSfcLayer.addTo(MAP);
  _synBtnOn("btn-conv-sfc");
}

function synToggleConv850() {
  _synConv850Show = !_synConv850Show;
  var MAP = _getMap(); if (!MAP) return;
  if (!_synConv850Show) {
    if (_synConv850Layer) MAP.removeLayer(_synConv850Layer);
    _synBtnOff("btn-conv-850"); return;
  }
  if (_synConv850Layer) MAP.removeLayer(_synConv850Layer);
  _synConv850Layer = L.layerGroup();
  _drawConvSegs((_CONV_DATA["850"] || {})[_synUAHourKey] || [], _synConv850Layer, false);
  _synConv850Layer.addTo(MAP);
  _synBtnOn("btn-conv-850");
}

(function(){
  var _orig = window.synUpdateTS;
  if (typeof _orig !== "function") return;
  window.synUpdateTS = function(ts) {
    _orig(ts);
    if (_synConvSfcShow) {
      var MAP = _getMap(); if (!MAP) return;
      if (_synConvSfcLayer) MAP.removeLayer(_synConvSfcLayer);
      _synConvSfcLayer = L.layerGroup();
      _drawConvSegs((_CONV_DATA.sfc || {})[ts] || [], _synConvSfcLayer, true);
      _synConvSfcLayer.addTo(MAP);
    }
    if (_synShowSfcTrough) {
      var MAP = _getMap(); if (!MAP) return;
      if (_synSfcTroughLayer) MAP.removeLayer(_synSfcTroughLayer);
      _synSfcTroughLayer = L.layerGroup();
      _drawTroughSegs((_CONV_DATA.sfc_trough || {})[ts] || [], _synSfcTroughLayer);
      _synSfcTroughLayer.addTo(MAP);
    }
  };
})();

(function(){
  var _orig = window.synFilterHour;
  if (typeof _orig !== "function") return;
  window.synFilterHour = function(h) {
    _orig(h);
    if (!_synConv850Show) return;
    var MAP = _getMap(); if (!MAP) return;
    if (_synConv850Layer) MAP.removeLayer(_synConv850Layer);
    _synConv850Layer = L.layerGroup();
    _drawConvSegs((_CONV_DATA["850"] || {})[_synUAHourKey] || [], _synConv850Layer, false);
    _synConv850Layer.addTo(MAP);
  };
})();
""" + "</script>\n"

m.get_root().html.add_child(Element(_conv_js))

print('✓ Convergence zones injected into map')
print(f'  SFC: {sum(len(v) for v in _conv_sfc_by_ts.values())} total segments across {len(_conv_sfc_by_ts)} timestamps')
print(f'  850: {sum(len(v) for v in _conv_850_by_hr.values())} total segments across {len(_conv_850_by_hr)} hours')




# ══════════════════════════════════════════════════════════════════════════════
# CELL: Upper Air Radiosonde Reporting Availability  — injected into main map
# Source: Colab availability cell adapted for github script
# ══════════════════════════════════════════════════════════════════════════════
try:
    import json as _ua_json

    df = ua_raw_df.copy()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _ua_fmt(val, spec=".1f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return format(val, spec)

    def _ua_safe(v):
        if v is None: return None
        if isinstance(v, float) and np.isnan(v): return None
        return float(v)

    def _ua_stats(sub):
        if sub is None or len(sub) == 0: return None
        return {
            "levels":    int(len(sub)),
            "temp_mean": float(sub["TEMP"].mean()),
            "wind_mean": float(sub["SPED"].mean()),
            "min_pres":  float(sub["PRES"].min()),
            "max_pres":  float(sub["PRES"].max()),
        }

    def _ua_profile(sub):
        if sub is None or len(sub) == 0: return [], [], []
        s = sub.sort_values("PRES", ascending=False)
        return (
            [_ua_safe(v) for v in s["PRES"]],
            [_ua_safe(v) for v in s["TEMP"]],
            [_ua_safe(v) for v in s["SPED"]],
        )

    def _ua_get_timestamp(sub):
        if sub is None or len(sub) == 0: return None, None
        for col in ["datetime", "valid_time", "time", "DATE", "date"]:
            if col in sub.columns:
                try:
                    t = pd.to_datetime(sub[col].iloc[0])
                    return t, t.strftime("%d %b %Y %HZ")
                except:
                    pass
        return None, None

    _UA_STATUS_SCORE = {"✔ ok": 2, "↻✔ recovered": 1, "✘ no data": 0}

    def _ua_status_color(st):
        s = _UA_STATUS_SCORE.get(st, 0)
        if s == 2: return "#27ae60"
        if s == 1: return "#2980b9"
        return "#e74c3c"

    def _ua_status_bg(st):
        s = _UA_STATUS_SCORE.get(st, 0)
        if s == 2: return "#f0faf4"
        if s == 1: return "#eef6fd"
        return "#fff0f0"

    # ── build station payload ─────────────────────────────────────────────────
    _ua_stations_data = []

    for _s in UPPER_AIR_STATIONS:
        sid = _s["id"]

        sub_a = df[(df["icao"] == sid) & (df["hour"] == 0)]
        sub_b = df[(df["icao"] == sid) & (df["hour"] == 12)]

        st_a = _status.get((sid, 0),  "✘ no data")
        st_b = _status.get((sid, 12), "✘ no data")

        t_a, lbl_a = _ua_get_timestamp(sub_a)
        t_b, lbl_b = _ua_get_timestamp(sub_b)

        if t_a is not None and t_b is not None and t_b < t_a:
            sub_L, sub_R = sub_b, sub_a
            st_L,  st_R  = st_b,  st_a
            lbl_L, lbl_R = lbl_b, lbl_a
        else:
            sub_L, sub_R = sub_a, sub_b
            st_L,  st_R  = st_a,  st_b
            lbl_L, lbl_R = lbl_a or "00Z", lbl_b or "12Z"

        d_L = _ua_stats(sub_L)
        d_R = _ua_stats(sub_R)
        p_L, t_L, w_L = _ua_profile(sub_L)
        p_R, t_R, w_R = _ua_profile(sub_R)
        c_L  = _ua_status_color(st_L)
        c_R  = _ua_status_color(st_R)
        bg_L = _ua_status_bg(st_L)
        bg_R = _ua_status_bg(st_R)
        cid  = sid.replace(" ", "_")

        def _ua_card_html(label, emoji, st, d, c, bg):
            levels = str(d['levels'])                                                      if d else '—'
            temp   = _ua_fmt(d['temp_mean']) + " °C"                                      if d else '—'
            wind   = _ua_fmt(d['wind_mean']) + " kt"                                      if d else '—'
            pres   = f"{_ua_fmt(d['min_pres'],'.0f')}–{_ua_fmt(d['max_pres'],'.0f')} hPa" if d else '—'
            return (
                f'<div style="flex:1;border:2px solid {c};border-radius:7px;background:{bg};overflow:hidden;">'
                f'<div style="background:{c};color:white;font-weight:bold;font-size:11px;padding:5px 8px;line-height:1.4;">'
                f'{emoji} {label}</div>'
                f'<div style="padding:7px 8px;font-size:12px;">'
                f'<div style="display:inline-block;padding:1px 7px;border-radius:10px;background:{c}22;'
                f'color:{c};font-weight:600;font-size:11px;margin-bottom:5px;">{st}</div>'
                f'<table style="border-collapse:collapse;width:100%;">'
                f'<tr><td style="color:#888;padding-right:8px;padding-bottom:2px;">Levels</td><td style="font-weight:500;">{levels}</td></tr>'
                f'<tr><td style="color:#888;padding-right:8px;padding-bottom:2px;">Temp</td><td style="font-weight:500;">{temp}</td></tr>'
                f'<tr><td style="color:#888;padding-right:8px;padding-bottom:2px;">Wind</td><td style="font-weight:500;">{wind}</td></tr>'
                f'<tr><td style="color:#888;padding-right:8px;padding-bottom:2px;">Pressure</td><td style="font-weight:500;">{pres}</td></tr>'
                f'</table></div></div>'
            )

        popup_html = (
            f'<div style="font-family:Arial,sans-serif;font-size:12px;width:620px;">'
            f'<div style="background:#1a1a2e;color:white;padding:8px 12px;'
            f'border-radius:6px 6px 0 0;font-weight:bold;font-size:13px;">'
            f'📡 {_s["name"]} '
            f'<span style="font-weight:normal;font-size:11px;opacity:0.75;margin-left:6px;">({sid})</span>'
            f'</div>'
            f'<div style="display:flex;gap:8px;padding:10px;background:#f8f9fa;'
            f'border:1px solid #ddd;border-top:none;">'
            f'{_ua_card_html(lbl_L, "🕛", st_L, d_L, c_L, bg_L)}'
            f'{_ua_card_html(lbl_R, "🕕", st_R, d_R, c_R, bg_R)}'
            f'</div>'
            f'<div style="background:#fff;border:1px solid #ddd;border-top:none;'
            f'border-radius:0 0 6px 6px;padding:10px;">'
            f'<div style="display:flex;gap:6px;margin-bottom:8px;">'
            f'<button onclick="uaAvailShowChart(\'{cid}\',\'skewt\')" id="uabtn-skewt-{cid}" '
            f'style="flex:1;padding:5px;border:1px solid #c0392b;border-radius:5px;'
            f'background:#c0392b;color:white;font-size:11px;cursor:pointer;font-weight:bold;">'
            f'🌡 Skew-T</button>'
            f'<button onclick="uaAvailShowChart(\'{cid}\',\'wind\')" id="uabtn-wind-{cid}" '
            f'style="flex:1;padding:5px;border:1px solid #aaa;border-radius:5px;'
            f'background:#f5f5f5;color:#333;font-size:11px;cursor:pointer;">'
            f'💨 Wind Profile</button>'
            f'</div>'
            f'<div id="uachart-skewt-{cid}" style="display:block;position:relative;">'
            f'<canvas id="uacanvas-skewt-{cid}" style="width:100%;height:480px;"></canvas></div>'
            f'<div id="uachart-wind-{cid}" style="display:none;">'
            f'<canvas id="uacanvas-wind-{cid}" style="width:100%;height:480px;"></canvas></div>'
            f'</div>'
            f'</div>'
        )

        _ua_stations_data.append({
            "lat": _s["lat"], "lon": _s["lon"],
            "c_L": c_L, "c_R": c_R,
            "cid": cid,
            "popup": popup_html,
            "lbl_L": lbl_L, "lbl_R": lbl_R,
            "p_L": p_L, "t_L": t_L, "w_L": w_L,
            "p_R": p_R, "t_R": t_R, "w_R": w_R,
        })

    _ua_stations_json = _ua_json.dumps(_ua_stations_data)

    # ── build the overlay panel HTML + JS ────────────────────────────────────
    _ua_avail_html = (
        # Chart.js CDN (safe to add twice — browsers deduplicate)
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>\n'

        # ── floating panel ──
        '<div id="ua-avail-panel" style="'
        'display:none;position:fixed;top:0;left:0;width:100%;height:100%;'
        'z-index:20000;background:rgba(0,0,0,0.55);overflow:auto;">'

        '<div style="'
        'background:#fff;margin:30px auto;border-radius:10px;'
        'max-width:960px;width:96%;font-family:Arial,sans-serif;'
        'box-shadow:0 8px 32px rgba(0,0,0,0.35);overflow:hidden;">'

        # header bar
        '<div style="background:#1a1a2e;color:#fff;padding:12px 18px;'
        'display:flex;align-items:center;justify-content:space-between;">'
        '<span style="font-weight:bold;font-size:15px;">📡 Upper Air Radiosonde Reporting Availability</span>'
        '<button onclick="synToggleUAAvail()" '
        'style="background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.4);'
        'border-radius:5px;color:#fff;font-size:13px;padding:4px 12px;cursor:pointer;">✕ Close</button>'
        '</div>'

        # legend strip
        '<div style="background:#f8f9fa;padding:8px 18px;border-bottom:1px solid #e0e0e0;'
        'display:flex;gap:16px;flex-wrap:wrap;font-size:11px;align-items:center;">'
        '<b style="color:#333;font-size:12px;">Legend:</b>'
        '<span style="color:#555;font-size:11px;font-style:italic;">Left half = earlier &nbsp;|&nbsp; Right half = later</span>'

        # SVG legend items
        '<span style="display:flex;align-items:center;gap:4px;">'
        '<svg width="18" height="18" viewBox="0 0 22 22">'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,0 0,20 Z" fill="#27ae60"/>'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,1 0,20 Z" fill="#27ae60"/>'
        '<circle cx="11" cy="11" r="10" fill="none" stroke="white" stroke-width="1.5"/>'
        '<line x1="11" y1="1" x2="11" y2="21" stroke="white" stroke-width="1.2"/></svg>Both OK</span>'

        '<span style="display:flex;align-items:center;gap:4px;">'
        '<svg width="18" height="18" viewBox="0 0 22 22">'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,0 0,20 Z" fill="#e74c3c"/>'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,1 0,20 Z" fill="#27ae60"/>'
        '<circle cx="11" cy="11" r="10" fill="none" stroke="white" stroke-width="1.5"/>'
        '<line x1="11" y1="1" x2="11" y2="21" stroke="white" stroke-width="1.2"/></svg>Earlier missing</span>'

        '<span style="display:flex;align-items:center;gap:4px;">'
        '<svg width="18" height="18" viewBox="0 0 22 22">'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,0 0,20 Z" fill="#27ae60"/>'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,1 0,20 Z" fill="#e74c3c"/>'
        '<circle cx="11" cy="11" r="10" fill="none" stroke="white" stroke-width="1.5"/>'
        '<line x1="11" y1="1" x2="11" y2="21" stroke="white" stroke-width="1.2"/></svg>Later missing</span>'

        '<span style="display:flex;align-items:center;gap:4px;">'
        '<svg width="18" height="18" viewBox="0 0 22 22">'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,0 0,20 Z" fill="#2980b9"/>'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,1 0,20 Z" fill="#27ae60"/>'
        '<circle cx="11" cy="11" r="10" fill="none" stroke="white" stroke-width="1.5"/>'
        '<line x1="11" y1="1" x2="11" y2="21" stroke="white" stroke-width="1.2"/></svg>One recovered</span>'

        '<span style="display:flex;align-items:center;gap:4px;">'
        '<svg width="18" height="18" viewBox="0 0 22 22">'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,0 0,20 Z" fill="#e74c3c"/>'
        '<path d="M 11,11 m 0,-10 a 10,10 0 0,1 0,20 Z" fill="#e74c3c"/>'
        '<circle cx="11" cy="11" r="10" fill="none" stroke="white" stroke-width="1.5"/>'
        '<line x1="11" y1="1" x2="11" y2="21" stroke="white" stroke-width="1.2"/></svg>Both missing</span>'
        '</div>'

        # map container
        '<div id="ua-avail-map" style="height:520px;width:100%;"></div>'

        '</div></div>\n'  # end panel

        # ── JS ────────────────────────────────────────────────────────────────
        f'<script>\n'
        f'var _UA_AVAIL_DATA = {_ua_stations_json};\n'
        f'var _uaAvailMap    = null;\n'
        f'var _uaAvailCharts = {{}};\n'
        f'var _uaAvailVisible = false;\n'
        '\n'
        'function synToggleUAAvail() {\n'
        '  _uaAvailVisible = !_uaAvailVisible;\n'
        '  var panel = document.getElementById("ua-avail-panel");\n'
        '  if (!panel) return;\n'
        '  panel.style.display = _uaAvailVisible ? "block" : "none";\n'
        '  var btn = document.getElementById("btn-ua-avail");\n'
        '  if (btn) {\n'
        '    btn.style.background = _uaAvailVisible ? "#4a7fc1" : "#1a3a6a";\n'
        '  }\n'
        '  if (_uaAvailVisible && !_uaAvailMap) {\n'
        '    setTimeout(_uaAvailInitMap, 80);\n'
        '  }\n'
        '}\n'
        '\n'
        'function _uaAvailInitMap() {\n'
        '  if (_uaAvailMap) return;\n'
        '  _uaAvailMap = L.map("ua-avail-map", {\n'
        '    center: [55, -100], zoom: 3,\n'
        '    preferCanvas: true\n'
        '  });\n'
        '  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {\n'
        '    attribution: "&copy; CartoDB",\n'
        '    subdomains: "abcd", maxZoom: 19\n'
        '  }).addTo(_uaAvailMap);\n'
        '\n'
        '  _UA_AVAIL_DATA.forEach(function(s) {\n'
        '    var svg = [\n'
        '      \'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">\',\n'
        '      \'<path d="M 12,12 m 0,-10 a 10,10 0 0,0 0,20 Z" fill="\' + s.c_L + \'"/>\',\n'
        '      \'<path d="M 12,12 m 0,-10 a 10,10 0 0,1 0,20 Z" fill="\' + s.c_R + \'"/>\',\n'
        '      \'<circle cx="12" cy="12" r="10" fill="none" stroke="white" stroke-width="1.5"/>\',\n'
        '      \'<line x1="12" y1="2" x2="12" y2="22" stroke="white" stroke-width="1.2"/>\',\n'
        '      \'</svg>\'\n'
        '    ].join("");\n'
        '    var icon = L.divIcon({html: svg, iconSize:[24,24], iconAnchor:[12,12], className:""});\n'
        '    var marker = L.marker([s.lat, s.lon], {icon: icon})\n'
        '      .bindPopup(s.popup, {maxWidth: 640, minWidth: 630});\n'
        '    marker.on("popupopen", function() {\n'
        '      setTimeout(function() { _uaAvailDrawCharts(s); }, 80);\n'
        '    });\n'
        '    marker.addTo(_uaAvailMap);\n'
        '  });\n'
        '}\n'
        '\n'
        'function uaAvailShowChart(cid, which) {\n'
        '  ["skewt","wind"].forEach(function(t) {\n'
        '    var el  = document.getElementById("uachart-" + t + "-" + cid);\n'
        '    var btn = document.getElementById("uabtn-" + t + "-" + cid);\n'
        '    var colors = {skewt:"#c0392b", wind:"#2980b9"};\n'
        '    if (el)  el.style.display = (t === which) ? "block" : "none";\n'
        '    if (btn) {\n'
        '      if (t === which) {\n'
        '        btn.style.background  = colors[t];\n'
        '        btn.style.borderColor = colors[t];\n'
        '        btn.style.color       = "white";\n'
        '        btn.style.fontWeight  = "bold";\n'
        '      } else {\n'
        '        btn.style.background  = "#f5f5f5";\n'
        '        btn.style.borderColor = "#aaa";\n'
        '        btn.style.color       = "#333";\n'
        '        btn.style.fontWeight  = "normal";\n'
        '      }\n'
        '    }\n'
        '  });\n'
        '}\n'
        '\n'
        'function _uaAvailDrawSkewT(s) {\n'
        '  var canvas = document.getElementById("uacanvas-skewt-" + s.cid);\n'
        '  if (!canvas) return;\n'
        '  var rect = canvas.getBoundingClientRect();\n'
        '  var W = rect.width || 580, H = rect.height || 480;\n'
        '  canvas.width = W; canvas.height = H;\n'
        '  var ctx = canvas.getContext("2d");\n'
        '  ctx.clearRect(0, 0, W, H);\n'
        '  var ml=52, mr=20, mt=30, mb=40;\n'
        '  var pw = W-ml-mr, ph = H-mt-mb;\n'
        '  var P_BOT=1050, P_TOP=100, T_MIN=-80, T_MAX=40, SKEW=0.5;\n'
        '  function yP(p) { return mt + ph*(Math.log(P_TOP)-Math.log(p))/(Math.log(P_TOP)-Math.log(P_BOT)); }\n'
        '  function xT(t,p) { return ml + pw*(t-T_MIN)/(T_MAX-T_MIN) + SKEW*(yP(P_BOT)-yP(p)); }\n'
        '  ctx.save();\n'
        '  ctx.beginPath(); ctx.rect(ml,mt,pw,ph); ctx.clip();\n'
        '  ctx.fillStyle="#fafafa"; ctx.fillRect(ml,mt,pw,ph);\n'
        '  ctx.strokeStyle="rgba(220,120,60,0.35)"; ctx.lineWidth=0.8; ctx.setLineDash([4,4]);\n'
        '  [240,260,280,300,320,340,360,380,400,420].forEach(function(theta) {\n'
        '    ctx.beginPath(); var first=true;\n'
        '    for (var p=P_BOT; p>=P_TOP; p-=5) {\n'
        '      var T=theta*Math.pow(p/1000,0.286)-273.15;\n'
        '      var x=xT(T,p), y=yP(p);\n'
        '      first?ctx.moveTo(x,y):ctx.lineTo(x,y); first=false;\n'
        '    }\n'
        '    ctx.stroke();\n'
        '  });\n'
        '  ctx.strokeStyle="rgba(100,100,200,0.3)"; ctx.lineWidth=0.8; ctx.setLineDash([]);\n'
        '  for (var T=T_MIN; T<=T_MAX; T+=10) {\n'
        '    ctx.beginPath(); ctx.moveTo(xT(T,P_BOT),yP(P_BOT)); ctx.lineTo(xT(T,P_TOP),yP(P_TOP)); ctx.stroke();\n'
        '  }\n'
        '  ctx.strokeStyle="rgba(0,100,200,0.6)"; ctx.lineWidth=1.5; ctx.setLineDash([6,3]);\n'
        '  ctx.beginPath(); ctx.moveTo(xT(0,P_BOT),yP(P_BOT)); ctx.lineTo(xT(0,P_TOP),yP(P_TOP)); ctx.stroke();\n'
        '  ctx.setLineDash([]);\n'
        '  [1000,925,850,700,500,400,300,250,200,150,100].forEach(function(p) {\n'
        '    if (p>P_BOT||p<P_TOP) return;\n'
        '    var y=yP(p);\n'
        '    ctx.strokeStyle="rgba(150,150,150,0.5)"; ctx.lineWidth=(p===500||p===850)?1.2:0.7;\n'
        '    ctx.beginPath(); ctx.moveTo(ml,y); ctx.lineTo(ml+pw,y); ctx.stroke();\n'
        '  });\n'
        '  ctx.restore();\n'
        '  ctx.fillStyle="#444"; ctx.font="10px Arial"; ctx.textAlign="right";\n'
        '  [1000,925,850,700,500,400,300,250,200,150,100].forEach(function(p) {\n'
        '    if (p>P_BOT||p<P_TOP) return;\n'
        '    ctx.fillText(p, ml-4, yP(p)+3);\n'
        '  });\n'
        '  ctx.fillStyle="rgba(80,80,160,0.7)"; ctx.font="9px Arial"; ctx.textAlign="center";\n'
        '  for (var TT=T_MIN; TT<=T_MAX; TT+=10) { ctx.fillText(TT+"°", xT(TT,P_BOT), mt+ph+12); }\n'
        '  ctx.save(); ctx.fillStyle="#333"; ctx.font="bold 10px Arial"; ctx.textAlign="center";\n'
        '  ctx.translate(12, mt+ph/2); ctx.rotate(-Math.PI/2); ctx.fillText("Pressure (hPa)", 0, 0);\n'
        '  ctx.restore();\n'
        '  ctx.fillStyle="#666"; ctx.font="10px Arial"; ctx.textAlign="center";\n'
        '  ctx.fillText("Temperature (°C)", ml+pw/2, mt+ph+28);\n'
        '  ctx.save(); ctx.beginPath(); ctx.rect(ml,mt,pw,ph); ctx.clip();\n'
        '  function drawSounding(pArr, tArr, color, dash) {\n'
        '    if (!pArr||pArr.length===0) return;\n'
        '    ctx.strokeStyle=color; ctx.lineWidth=2.2; ctx.setLineDash(dash||[]);\n'
        '    ctx.beginPath(); var first=true;\n'
        '    for (var i=0;i<pArr.length;i++) {\n'
        '      var p=pArr[i], t=tArr[i];\n'
        '      if (p===null||t===null){first=true;continue;}\n'
        '      if (p<P_TOP||p>P_BOT){first=true;continue;}\n'
        '      var x=xT(t,p), y=yP(p);\n'
        '      first?ctx.moveTo(x,y):ctx.lineTo(x,y); first=false;\n'
        '    }\n'
        '    ctx.stroke(); ctx.fillStyle=color; ctx.setLineDash([]);\n'
        '    for (var j=0;j<pArr.length;j++) {\n'
        '      var pp=pArr[j], tt=tArr[j];\n'
        '      if (pp===null||tt===null) continue;\n'
        '      if (pp<P_TOP||pp>P_BOT) continue;\n'
        '      ctx.beginPath(); ctx.arc(xT(tt,pp),yP(pp),2.5,0,2*Math.PI); ctx.fill();\n'
        '    }\n'
        '  }\n'
        '  drawSounding(s.p_L, s.t_L, "#e74c3c", []);\n'
        '  drawSounding(s.p_R, s.t_R, "#e67e22", [6,3]);\n'
        '  ctx.restore();\n'
        '  var lx=ml+pw-5, ly=mt+8;\n'
        '  [[s.lbl_L,"#e74c3c",[]],[s.lbl_R,"#e67e22",[5,3]]].forEach(function(item,i) {\n'
        '    var yy=ly+i*18; ctx.setLineDash(item[2]);\n'
        '    ctx.strokeStyle=item[1]; ctx.lineWidth=2;\n'
        '    ctx.beginPath(); ctx.moveTo(lx-52,yy+4); ctx.lineTo(lx-36,yy+4); ctx.stroke();\n'
        '    ctx.setLineDash([]);\n'
        '    ctx.fillStyle="#333"; ctx.font="10px Arial"; ctx.textAlign="right";\n'
        '    ctx.fillText(item[0], lx, yy+7);\n'
        '  });\n'
        '  ctx.fillStyle="#1a1a2e"; ctx.font="bold 11px Arial"; ctx.textAlign="center";\n'
        '  ctx.fillText("Skew-T Log-P  —  Temperature Profile", ml+pw/2, mt-8);\n'
        '}\n'
        '\n'
        'function _uaAvailDrawWind(s) {\n'
        '  var key = "wind-" + s.cid;\n'
        '  if (_uaAvailCharts[key]) { _uaAvailCharts[key].destroy(); delete _uaAvailCharts[key]; }\n'
        '  var ctxW = document.getElementById("uacanvas-wind-" + s.cid);\n'
        '  if (!ctxW) return;\n'
        '  function pts(xs, ys) {\n'
        '    var out=[];\n'
        '    for (var i=0;i<xs.length;i++) { if (xs[i]!==null&&ys[i]!==null) out.push({x:xs[i],y:ys[i]}); }\n'
        '    return out;\n'
        '  }\n'
        '  _uaAvailCharts[key] = new Chart(ctxW, {\n'
        '    type: "scatter",\n'
        '    data: {\n'
        '      datasets: [\n'
        '        { label: s.lbl_L + " (earlier)", data: pts(s.w_L, s.p_L),\n'
        '          borderColor:"#2980b9", backgroundColor:"#2980b944",\n'
        '          showLine:true, tension:0.3, pointRadius:3, borderWidth:2 },\n'
        '        { label: s.lbl_R + " (later)", data: pts(s.w_R, s.p_R),\n'
        '          borderColor:"#8e44ad", backgroundColor:"#8e44ad44",\n'
        '          showLine:true, tension:0.3, pointRadius:3, borderWidth:2, borderDash:[4,3] }\n'
        '      ]\n'
        '    },\n'
        '    options: {\n'
        '      responsive:true, animation:false,\n'
        '      plugins: {\n'
        '        legend: { position:"top", labels:{ boxWidth:12, font:{size:11} } },\n'
        '        tooltip: { callbacks: { label: function(c) {\n'
        '          return c.dataset.label+": "+c.parsed.x.toFixed(1)+" kt  |  "+c.parsed.y.toFixed(0)+" hPa";\n'
        '        }}}\n'
        '      },\n'
        '      scales: {\n'
        '        x: { title:{display:true,text:"Wind Speed (kt)",font:{size:11}}, grid:{color:"#eee"} },\n'
        '        y: { title:{display:true,text:"Pressure (hPa)",font:{size:11}},\n'
        '             reverse:true, grid:{color:"#eee"}, ticks:{font:{size:10}} }\n'
        '      }\n'
        '    }\n'
        '  });\n'
        '}\n'
        '\n'
        'function _uaAvailDrawCharts(s) { _uaAvailDrawSkewT(s); _uaAvailDrawWind(s); }\n'
        '</script>\n'
    )

    m.get_root().html.add_child(Element(_ua_avail_html))
    print(f'✓ UA Radiosonde Availability panel injected ({len(_ua_stations_data)} stations)')

except Exception as _ua_avail_ex:
    print(f'⚠ UA Availability panel skipped: {_ua_avail_ex}')

# ── save ──────────────────────────────────────────────────────────────────────
import os
os.makedirs('output', exist_ok=True)
out_path = 'output/synoptic_map.html'
m.save(out_path)

with open(out_path) as _f:
    _html = _f.read()

# Inject no-match warning div into the saved HTML body
_no_match_div = (
    '<div id="syn-no-match-msg" style="'
    'display:none;position:fixed;top:50px;left:10px;z-index:10000;'
    'background:rgba(220,30,30,0.92);color:#fff;border-radius:6px;'
    'padding:5px 14px;font-family:Courier New,monospace;font-size:11px;'
    'box-shadow:0 2px 8px rgba(0,0,0,0.2);pointer-events:none;'
    '"></div>'
)
_html = _html.replace('</body>', _no_match_div + '</body>')

# Write the patched HTML back to disk
with open(out_path, 'w') as _f:
    _f.write(_html)

print(f'✓ Map saved → {out_path}')


#####################################################
