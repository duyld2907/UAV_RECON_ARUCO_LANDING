import threading
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Image
from std_msgs.msg import Bool, String

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'drone_map_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>UAV Mission Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>

    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background: #070b10;
            color: #c9d1d9;
            font-family: "Courier New", monospace;
            overflow: hidden;
        }

        #topbar {
            height: 42px;
            background: #111820;
            border-bottom: 1px solid #30363d;
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 0 14px;
        }

        #topbar h1 {
            color: #58a6ff;
            font-size: 15px;
            letter-spacing: 0.5px;
        }

        #layer-select {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
        }

        #mode-info { color: #ffa657; font-size: 12px; }
        #mission-info { color: #79c0ff; font-size: 12px; }
        #gps-info { color: #3fb950; font-size: 12px; margin-left: auto; }

        #main-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 430px;
            height: calc(100vh - 42px);
            width: 100%;
        }

        #left-side {
            display: grid;
            grid-template-rows: minmax(0, 1fr) 56px;
            min-width: 0;
            height: 100%;
            border-right: 1px solid #30363d;
        }

        #map-wrap {
            position: relative;
            min-width: 0;
            min-height: 0;
        }

        #map {
            width: 100%;
            height: 100%;
        }

        #right-side {
            display: grid;
            grid-template-rows: 310px minmax(0, 1fr);
            height: 100%;
            background: #0d1117;
        }

        #cam-panel {
            border-bottom: 1px solid #30363d;
            background: #000;
            position: relative;
        }

        #cam-title {
            height: 28px;
            background: #0d419d;
            color: white;
            font-size: 12px;
            padding: 6px 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        #cam-img {
            width: 100%;
            height: calc(100% - 28px);
            object-fit: contain;
            background: #000;
            display: none;
        }

        #cam-no-signal {
            position: absolute;
            top: 54%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #444;
            font-size: 13px;
            text-align: center;
        }

        #log-panel {
            display: grid;
            grid-template-rows: 34px 92px minmax(0, 1fr);
            min-height: 0;
        }

        #log-title {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 8px 10px;
            color: #58a6ff;
            font-weight: bold;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
        }

        #telemetry-box {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: #30363d;
            border-bottom: 1px solid #30363d;
        }

        .telemetry-item {
            background: #0d1117;
            padding: 7px 9px;
            font-size: 11px;
        }

        .telemetry-label {
            color: #8b949e;
            margin-bottom: 3px;
        }

        .telemetry-value {
            color: #f0f6fc;
            font-weight: bold;
        }

        #flight-log {
            overflow-y: auto;
            padding: 10px;
            font-size: 12px;
            line-height: 1.45;
        }

        .log-line {
            border-left: 3px solid #30363d;
            padding: 5px 8px;
            margin-bottom: 7px;
            background: rgba(255,255,255,0.03);
            border-radius: 3px;
            word-break: break-word;
        }

        .log-time {
            color: #8b949e;
            margin-right: 6px;
        }

        .log-info { border-left-color: #58a6ff; }
        .log-ok   { border-left-color: #3fb950; }
        .log-warn { border-left-color: #ffa657; }
        .log-err  { border-left-color: #f85149; }
        .log-aruco { border-left-color: #d2a8ff; }

        #controlbar {
            background: #0d1117;
            padding: 8px 12px;
            display: flex;
            gap: 8px;
            align-items: center;
            border-top: 1px solid #30363d;
            min-width: 0;
        }

        .btn {
            padding: 8px 18px;
            border: none;
            border-radius: 4px;
            color: white;
            font-family: "Courier New", monospace;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
        }

        .btn:hover { opacity: 0.85; }

        #btn-takeoff { background: #2ea043; }
        #btn-start { background: #1f6feb; }
        #btn-land { background: #da3633; }
        #btn-clear { background: #9e6a03; }
        #btn-setwp { background: #8957e5; }

        .coord-input {
            width: 125px;
            background: #161b22;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 7px 8px;
            font-family: "Courier New", monospace;
            font-size: 12px;
        }

        #status {
            margin-left: auto;
            color: #3fb950;
            font-size: 12px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }

        #wp-count {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(13,17,23,0.88);
            color: #f85149;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: bold;
            z-index: 1000;
        }

        .uav-icon {
            width: 26px;
            height: 26px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.7));
        }

        .person-dot {
            width: 20px;
            height: 20px;
            background: #ff7b00;
            border: 2px solid white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
            box-shadow: 0 0 8px rgba(255,123,0,0.75);
        }

        .aruco-dot {
            width: 22px;
            height: 22px;
            background: #000;
            border: 2px solid #d2a8ff;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            box-shadow: 0 0 8px rgba(210,168,255,0.75);
        }

        .aruco-dot span:nth-child(1),
        .aruco-dot span:nth-child(4) { background: #fff; }
        .aruco-dot span:nth-child(2),
        .aruco-dot span:nth-child(3) { background: #111; }

        .wp-icon {
            width: 22px;
            height: 22px;
            background: #f85149;
            border: 2px solid #fff;
            border-radius: 50%;
            color: white;
            font-size: 9px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.65);
        }

        .leaflet-tooltip.marker-label {
            background: rgba(0,0,0,0.78);
            color: #fff;
            border: none;
            border-radius: 2px;
            padding: 3px 5px;
            font-size: 10px;
            box-shadow: none;
        }

        .leaflet-tooltip.marker-label::before { display: none; }

        @media (max-width: 1050px) {
            #main-layout { grid-template-columns: minmax(0, 1fr) 360px; }
            #right-side { grid-template-rows: 250px minmax(0, 1fr); }
            .coord-input { width: 100px; }
            .btn { padding: 8px 12px; }
        }
    </style>
</head>

<body>
<div id="topbar">
    <h1>▲ UAV MAP MONITOR</h1>
    <select id="layer-select" onchange="changeLayer(this.value)">
        <option value="satellite">Satellite (Esri)</option>
        <option value="street">Street Map</option>
        <option value="topo">Topo Map</option>
    </select>
    <span id="mode-info">Mode: --</span>
    <span id="mission-info">Mission: waiting...</span>
    <span id="gps-info">GPS: waiting...</span>
</div>

<div id="main-layout">
    <div id="left-side">
        <div id="map-wrap">
            <div id="map"></div>
            <div id="wp-count">Waypoints: 0</div>
        </div>

        <div id="controlbar">
            <button class="btn" id="btn-takeoff" onclick="sendCmd('takeoff')">▲ TAKEOFF</button>
            <button class="btn" id="btn-start" onclick="sendCmd('start')">▶ START</button>
            <button class="btn" id="btn-land" onclick="sendCmd('land')">▼ LAND</button>
            <button class="btn" id="btn-clear" onclick="clearWaypoints()">✕ CLEAR</button>

            <input class="coord-input" id="manual-lat" type="number" step="0.0000001" placeholder="WP lat">
            <input class="coord-input" id="manual-lng" type="number" step="0.0000001" placeholder="WP lon">
            <button class="btn" id="btn-setwp" onclick="setManualWP1()">SET WP1</button>

            <span id="status">Monitor ready</span>
        </div>
    </div>

    <div id="right-side">
        <div id="cam-panel">
            <div id="cam-title">
                <span>📷 CAMERA LIVE - PERSON / ARUCO</span>
                <span id="cam-state">NO SIGNAL</span>
            </div>
            <img id="cam-img" src="" alt="">
            <div id="cam-no-signal">NO CAMERA<br>SIGNAL</div>
        </div>

        <div id="log-panel">
            <div id="log-title">
                <span>FLIGHT LOG</span>
                <span id="log-count">0</span>
            </div>

            <div id="telemetry-box">
                <div class="telemetry-item">
                    <div class="telemetry-label">CURRENT GPS</div>
                    <div class="telemetry-value" id="cur-gps">--</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">ALTITUDE</div>
                    <div class="telemetry-value" id="cur-alt">-- m</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">MISSION STATE</div>
                    <div class="telemetry-value" id="cur-mission">--</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">ARUCO STATUS</div>
                    <div class="telemetry-value" id="cur-aruco">--</div>
                </div>
            </div>

            <div id="flight-log"></div>
        </div>
    </div>
</div>

<script>
const tileLayers = {
    satellite: L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        { attribution: 'Esri World Imagery', maxZoom: 19 }
    ),
    street: L.tileLayer(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        { attribution: 'OpenStreetMap', maxZoom: 19 }
    ),
    topo: L.tileLayer(
        'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        { attribution: 'OpenTopoMap', maxZoom: 17 }
    )
};

const map = L.map('map', {
    center: [10.8535, 106.7719],
    zoom: 17,
    layers: [tileLayers.satellite]
});

let currentLayer = 'satellite';
function changeLayer(type) {
    map.removeLayer(tileLayers[currentLayer]);
    map.addLayer(tileLayers[type]);
    currentLayer = type;
}

const droneIcon = L.divIcon({
    className: '',
    html: `
    <svg class="uav-icon" viewBox="0 0 32 32">
        <path d="M16 2 L24 29 L16 23 L8 29 Z"
              fill="#19a7ff"
              stroke="#ffffff"
              stroke-width="1.8"
              stroke-linejoin="round"/>
        <circle cx="16" cy="16" r="3.2" fill="#ffffff" opacity="0.95"/>
    </svg>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -15]
});

const personIcon = L.divIcon({
    className: '',
    html: `<div class="person-dot">人</div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    popupAnchor: [0, -12]
});

const arucoIcon = L.divIcon({
    className: '',
    html: `<div class="aruco-dot"><span></span><span></span><span></span><span></span></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
    popupAnchor: [0, -13]
});

let droneMarker = null;
let personMarker = null;
let arucoMarker = null;

let droneTrail = [];
let trailLine = null;
let activeLegLine = null;

let mapCentered = false;
let waypoints = [];
let wpMarkers = [];
let wpPolyline = null;

let lastGps = null;
let lastLogGpsTime = 0;
let logIndex = 0;
let lastMissionState = '';
let lastArucoStatus = '';

function nowText() {
    return new Date().toLocaleTimeString();
}

function addLog(msg, type='info') {
    logIndex += 1;
    const box = document.getElementById('flight-log');
    const line = document.createElement('div');
    line.className = 'log-line log-' + type;
    line.innerHTML = '<span class="log-time">[' + nowText() + ']</span>' + msg;
    box.appendChild(line);
    while (box.children.length > 250) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
    document.getElementById('log-count').textContent = String(logIndex);
}

function haversineM(a, b) {
    const R = 6371000;
    const lat1 = a[0] * Math.PI / 180;
    const lat2 = b[0] * Math.PI / 180;
    const dlat = (b[0] - a[0]) * Math.PI / 180;
    const dlon = (b[1] - a[1]) * Math.PI / 180;
    const x =
        Math.sin(dlat/2) * Math.sin(dlat/2) +
        Math.cos(lat1) * Math.cos(lat2) *
        Math.sin(dlon/2) * Math.sin(dlon/2);
    return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1-x));
}

function makeWpIcon(idx) {
    return L.divIcon({
        className: '',
        html: `<div class="wp-icon">W${idx}</div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
    });
}

function redrawPlannedRoute() {
    if (wpPolyline) map.removeLayer(wpPolyline);
    if (waypoints.length > 1) {
        wpPolyline = L.polyline(waypoints, {
            color: '#f85149',
            weight: 2,
            opacity: 0.9
        }).addTo(map);
    }
    document.getElementById('wp-count').textContent = 'Waypoints: ' + waypoints.length;
}

function drawWaypoint(lat, lng) {
    waypoints.push([lat, lng]);
    const idx = waypoints.length;

    const m = L.marker([lat, lng], { icon: makeWpIcon(idx), zIndexOffset: 800 }).addTo(map);
    m.bindTooltip('WP' + idx, {
        permanent: true,
        direction: 'top',
        offset: [0, -12],
        className: 'marker-label'
    });
    wpMarkers.push(m);

    redrawPlannedRoute();
    addLog('Thêm WP' + idx + ': lat=' + lat.toFixed(7) + ' lon=' + lng.toFixed(7), 'info');

    document.getElementById('status').textContent =
        '[WP' + idx + '] lat=' + lat.toFixed(7) + ' lon=' + lng.toFixed(7);
}

function clearWaypointsLocal() {
    waypoints = [];
    wpMarkers.forEach(m => map.removeLayer(m));
    wpMarkers = [];
    if (wpPolyline) { map.removeLayer(wpPolyline); wpPolyline = null; }
    if (activeLegLine) { map.removeLayer(activeLegLine); activeLegLine = null; }
    document.getElementById('wp-count').textContent = 'Waypoints: 0';
}

map.on('click', function(e) {
    addWaypoint(e.latlng.lat, e.latlng.lng);
});

function addWaypoint(lat, lng) {
    drawWaypoint(lat, lng);
    fetch('/waypoint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng })
    });
}

function clearWaypoints() {
    clearWaypointsLocal();
    addLog('Xóa waypoint hiển thị trên web', 'warn');
    document.getElementById('status').textContent = 'Cleared';
    fetch('/clear', { method: 'POST' });
}

function setManualWP1() {
    const lat = parseFloat(document.getElementById('manual-lat').value);
    const lng = parseFloat(document.getElementById('manual-lng').value);

    if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
        alert('Nhap day du lat/lon hop le. Vi du: 10.844926 va 106.796789');
        return;
    }
    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        alert('Lat phai nam trong [-90,90], lon phai nam trong [-180,180].');
        return;
    }

    clearWaypointsLocal();
    drawWaypoint(lat, lng);
    map.flyTo([lat, lng], Math.max(map.getZoom(), 18), { duration: 0.8 });

    fetch('/set_wp1', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng })
    });
}

function sendCmd(cmd) {
    fetch('/cmd/' + cmd, { method: 'POST' });
    document.getElementById('status').textContent = '[CMD] ' + cmd.toUpperCase() + ' sent';

    if (cmd === 'takeoff') {
        addLog('Gửi TAKEOFF. Web chỉ monitor, không tự ghi HOME và không tự quyết định mission.', 'warn');
    } else if (cmd === 'start') {
        addLog('Gửi START nếu node bay của bạn còn dùng. Log bay không phụ thuộc nút START.', 'warn');
    } else if (cmd === 'land') {
        addLog('Gửi LAND / Precision Land command.', 'warn');
    }
}

function updateActiveLegFromMission(state) {
    if (!lastGps || waypoints.length === 0) return;

    const s = String(state || '').toUpperCase();
    let idx = -1;

    let m = s.match(/WP\\s*([0-9]+)/);
    if (!m) m = s.match(/WAYPOINT\\s*([0-9]+)/);
    if (m) idx = parseInt(m[1]) - 1;

    if (idx >= 0 && idx < waypoints.length) {
        if (activeLegLine) map.removeLayer(activeLegLine);
        activeLegLine = L.polyline([lastGps, waypoints[idx]], {
            color: '#ffd33d',
            weight: 2,
            opacity: 0.95,
            dashArray: '8,6'
        }).addTo(map);

        const dist = haversineM(lastGps, waypoints[idx]);
        document.getElementById('status').textContent =
            'Đang bay tới WP' + (idx + 1) + ' | distance=' + dist.toFixed(1) + 'm';
    }
}

const socket = io();

socket.on('drone_gps', function(data) {
    const latlng = [data.lat, data.lon];
    lastGps = latlng;

    if (!droneMarker) {
        droneMarker = L.marker(latlng, { icon: droneIcon, zIndexOffset: 1000 }).addTo(map);
        droneMarker.bindTooltip('', {
            permanent: true,
            direction: 'top',
            offset: [0, -16],
            className: 'marker-label'
        });

        if (!mapCentered) {
            mapCentered = true;
            map.flyTo(latlng, 18, { duration: 1.2 });
        }

        addLog('Nhận GPS drone đầu tiên: lat=' + data.lat.toFixed(7) +
               ' lon=' + data.lon.toFixed(7) + ' alt=' + data.alt.toFixed(1) + 'm', 'ok');
    } else {
        droneMarker.setLatLng(latlng);
    }

    droneMarker.setTooltipContent('Drone<br>ALT ' + data.alt.toFixed(1) + 'm');

    droneTrail.push(latlng);
    if (droneTrail.length > 900) droneTrail.shift();

    if (trailLine) map.removeLayer(trailLine);
    if (droneTrail.length > 1) {
        trailLine = L.polyline(droneTrail, {
            color: '#00d084',
            weight: 2,
            opacity: 0.85
        }).addTo(map);
    }

    document.getElementById('gps-info').textContent =
        'lat=' + data.lat.toFixed(6) +
        ' lon=' + data.lon.toFixed(6) +
        ' alt=' + data.alt.toFixed(1) + 'm';

    document.getElementById('cur-gps').textContent =
        data.lat.toFixed(6) + ', ' + data.lon.toFixed(6);
    document.getElementById('cur-alt').textContent = data.alt.toFixed(1) + ' m';

    const now = Date.now();
    if (now - lastLogGpsTime > 1500) {
        lastLogGpsTime = now;
        let msg = 'GPS hiện tại: lat=' + data.lat.toFixed(7) +
                  ' lon=' + data.lon.toFixed(7) +
                  ' alt=' + data.alt.toFixed(1) + 'm';

        if (lastMissionState) msg = '[' + lastMissionState + '] ' + msg;
        addLog(msg, 'info');
    }

    updateActiveLegFromMission(lastMissionState);
});

socket.on('flight_mode', function(data) {
    document.getElementById('mode-info').textContent = 'Mode: ' + data.mode;
});

socket.on('mission_state', function(data) {
    const state = data.state || '';
    if (state === lastMissionState) return;
    lastMissionState = state;

    document.getElementById('mission-info').textContent = 'Mission: ' + state;
    document.getElementById('cur-mission').textContent = state;

    const s = state.toUpperCase();
    let type = 'info';

    if (s.includes('WP') || s.includes('WAYPOINT') || s.includes('GOTO')) type = 'ok';
    if (s.includes('HOME') || s.includes('RTL') || s.includes('RETURN')) type = 'warn';
    if (s.includes('ARUCO') || s.includes('MARKER') || s.includes('ALIGN') || s.includes('LOCK')) type = 'aruco';
    if (s.includes('LAND')) type = 'warn';

    addLog('Mission update: ' + state, type);
    updateActiveLegFromMission(state);
});

socket.on('aruco_status', function(data) {
    const status = data.status || '';
    if (status === lastArucoStatus) return;
    lastArucoStatus = status;

    document.getElementById('cur-aruco').textContent = status.substring(0, 36);

    const s = status.toUpperCase();
    let label = 'AruCo';

    if (s.includes('BIG') || s.includes('LARGE') || s.includes('ALIGN_BIG') || s.includes('LOCK_BIG')) {
        label = 'Đang bắt mã lớn';
    } else if (s.includes('SMALL') || s.includes('ALIGN_SMALL') || s.includes('LOCK_SMALL')) {
        label = 'Đang bắt mã nhỏ';
    } else if (s.includes('LAND')) {
        label = 'Đang landing theo ArUco';
    }

    addLog(label + ' | ' + status, 'aruco');
});

socket.on('aruco_gps', function(data) {
    const latlng = [data.lat, data.lon];

    if (!arucoMarker) {
        arucoMarker = L.marker(latlng, {
            icon: arucoIcon,
            zIndexOffset: 960
        }).addTo(map);

        arucoMarker.bindTooltip('', {
            permanent: true,
            direction: 'top',
            offset: [0, -13],
            className: 'marker-label'
        });
    } else {
        arucoMarker.setLatLng(latlng);
    }

    arucoMarker.setTooltipContent('AruCo<br>' + data.lat.toFixed(6) + ', ' + data.lon.toFixed(6));
    addLog('AruCo GPS: lat=' + data.lat.toFixed(7) + ' lon=' + data.lon.toFixed(7), 'aruco');
});

socket.on('detected_object', function(data) {
    const latlng = [data.lat, data.lon];

    if (!personMarker) {
        personMarker = L.marker(latlng, {
            icon: personIcon,
            zIndexOffset: 900
        }).addTo(map);

        personMarker.bindTooltip('', {
            permanent: true,
            direction: 'top',
            offset: [0, -12],
            className: 'marker-label'
        });
    } else {
        personMarker.setLatLng(latlng);
    }

    personMarker.setTooltipContent('Person<br>' + data.lat.toFixed(6) + ', ' + data.lon.toFixed(6));
    personMarker.bindPopup('Person detected<br>lat=' + data.lat.toFixed(7) + '<br>lon=' + data.lon.toFixed(7));

    addLog('Detect người: lat=' + data.lat.toFixed(7) + ' lon=' + data.lon.toFixed(7), 'ok');
});

socket.on('camera_frame', function(data) {
    const img = document.getElementById('cam-img');
    img.src = 'data:image/jpeg;base64,' + data.frame;
    img.style.display = 'block';
    document.getElementById('cam-no-signal').style.display = 'none';
    document.getElementById('cam-state').textContent =
        data.source ? ('LIVE ' + data.source) : 'LIVE';
});

socket.on('camera_status', function(data) {
    addLog('Camera source active: ' + data.source, 'ok');
    document.getElementById('cam-state').textContent = 'LIVE ' + data.source;
});

socket.on('web_log', function(data) {
    const level = data.level || 'info';
    const msg = data.msg || '';
    addLog(msg, level);
});

socket.on('status', function(data) {
    document.getElementById('status').textContent = data.msg;
    addLog(data.msg, 'info');
});

addLog('Web monitor ready. Mission log sẽ đi theo /mission_state và /aruco_status, không phụ thuộc nút START.', 'ok');
</script>
</body>
</html>
"""


class WebMapNode(Node):

    def __init__(self):
        super().__init__('web_map_node')
        self.get_logger().info("Web Map Node started")

        # Camera web optimization
        self.declare_parameter('cam_fps', 8.0)
        self.declare_parameter('cam_width', 320)
        self.declare_parameter('cam_height', 240)
        self.declare_parameter('jpeg_quality', 50)
        # Chỉ xoay những topic raw bị ngược để web live nhìn thẳng.
        # Các topic debug như /camera/image_debug hoặc /aruco/debug_image đã được node tương ứng xoay riêng,
        # nên không đưa vào danh sách này để tránh xoay hai lần.
        self.declare_parameter('camera_rotate_180', True)
        self.declare_parameter(
            'camera_rotate_topics_csv',
            '/camera/bottom/image_raw,/cam/bottom/image_raw,/cam/front/image_raw,/image_raw,/usb_cam/image_raw'
        )

        self.last_cam_emit = 0.0
        self.cam_fps = max(0.5, float(self.get_parameter('cam_fps').value))
        cam_w = max(160, int(self.get_parameter('cam_width').value))
        cam_h = max(120, int(self.get_parameter('cam_height').value))
        self.cam_size = (cam_w, cam_h)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.jpeg_quality = max(20, min(95, self.jpeg_quality))
        self.camera_rotate_180 = bool(self.get_parameter('camera_rotate_180').value)
        self.camera_rotate_topics = set(
            t.strip() for t in str(self.get_parameter('camera_rotate_topics_csv').value).split(',') if t.strip()
        )

        # Publishers - giữ lại để không phá workflow cũ.
        self.gps_pub = self.create_publisher(NavSatFix, '/target_gps', 10)
        self.takeoff_pub = self.create_publisher(Bool, '/drone_takeoff', 10)
        self.start_pub = self.create_publisher(Bool, '/start_trajectory', 10)
        self.land_pub = self.create_publisher(Bool, '/drone_land', 10)
        self.clear_pub = self.create_publisher(Bool, '/clear_waypoints', 10)

        # Subscribers cũ.
        self.create_subscription(NavSatFix, '/drone_gps', self._cb_gps, 10)
        self.create_subscription(String, '/flight_mode', self._cb_mode, 10)
        # Camera: subscribe nhiều topic thường gặp.
        # Topic nào có frame đầu tiên sẽ được chọn làm nguồn camera live.
        self.active_camera_topic = None
        self.camera_topics = [
            # Ưu tiên debug AI đã xoay thẳng trong AL_node.
            '/camera/image_debug',
            # Nếu ArUco debug được bật, topic này cũng đã xoay thẳng trong aruco_node.
            '/aruco/debug_image',
            # Fallback raw từ camera đáy; web_map sẽ tự xoay 180 độ cho topic này.
            '/camera/bottom/image_raw',
            '/cam/bottom/image_raw',
            '/cam/front/image_raw',
            '/image_raw',
            '/usb_cam/image_raw'
        ]
        for topic in self.camera_topics:
            self.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self._cb_camera(msg, t),
                10
            )

        self.create_subscription(NavSatFix, '/detected_object_gps', self._cb_detected, 10)

        # Optional subscribers cho log bay tự động.
        # Nếu mission node chưa publish thì web vẫn chạy bình thường.
        self.create_subscription(String, '/mission_state', self._cb_mission_state, 10)
        self.create_subscription(String, '/aruco_status', self._cb_aruco_status, 10)
        self.create_subscription(String, '/web_log', self._cb_web_log, 10)
        self.create_subscription(NavSatFix, '/aruco_gps', self._cb_aruco_gps, 10)

    def _cb_gps(self, msg: NavSatFix):
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return
        socketio.emit('drone_gps', {
            'lat': msg.latitude,
            'lon': msg.longitude,
            'alt': msg.altitude
        })

    def _cb_mode(self, msg: String):
        socketio.emit('flight_mode', {'mode': msg.data})

    def _cb_mission_state(self, msg: String):
        socketio.emit('mission_state', {'state': msg.data})

    def _cb_aruco_status(self, msg: String):
        socketio.emit('aruco_status', {'status': msg.data})

    def _cb_web_log(self, msg: String):
        socketio.emit('web_log', {
            'msg': msg.data,
            'level': 'info'
        })

    def _cb_aruco_gps(self, msg: NavSatFix):
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return
        socketio.emit('aruco_gps', {
            'lat': msg.latitude,
            'lon': msg.longitude,
            'alt': msg.altitude
        })

    def _cb_camera(self, msg: Image, source_topic: str):
        try:
            # Chọn nguồn camera đầu tiên có frame.
            # Nếu đang có nguồn active thì bỏ qua các nguồn khác để tránh giật/đổi camera liên tục.
            if self.active_camera_topic is None:
                self.active_camera_topic = source_topic
                self.get_logger().info(f'Camera source selected: {source_topic}')
                socketio.emit('camera_status', {'source': source_topic})
            elif source_topic != self.active_camera_topic:
                return

            now = time.time()
            if now - self.last_cam_emit < 1.0 / self.cam_fps:
                return
            self.last_cam_emit = now

            if msg.encoding in ('rgb8', 'bgr8'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                if msg.encoding == 'rgb8':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ('mono8', '8UC1'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return

            # Chỉ xoay ở nhánh web live để quan sát cho dễ.
            # Không ảnh hưởng /camera/bottom/image_raw, ArUco, AI, PID hoặc Kalman.
            if self.camera_rotate_180 and source_topic in self.camera_rotate_topics:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame = cv2.resize(frame, self.cam_size, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                return

            socketio.emit('camera_frame', {
                'frame': base64.b64encode(buf).decode('utf-8'),
                'source': source_topic
            })

        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def _cb_detected(self, msg: NavSatFix):
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return
        socketio.emit('detected_object', {
            'lat': msg.latitude,
            'lon': msg.longitude
        })

    def pub_waypoint(self, lat, lon):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0
        self.gps_pub.publish(msg)

    def pub_takeoff(self):
        msg = Bool()
        msg.data = True
        self.takeoff_pub.publish(msg)

    def pub_start(self):
        msg = Bool()
        msg.data = True
        self.start_pub.publish(msg)

    def pub_land(self):
        msg = Bool()
        msg.data = True
        self.land_pub.publish(msg)

    def pub_clear_waypoints(self):
        msg = Bool()
        msg.data = True
        self.clear_pub.publish(msg)


ros_node = None


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/waypoint', methods=['POST'])
def waypoint():
    data = request.json
    if ros_node:
        ros_node.pub_waypoint(float(data['lat']), float(data['lng']))
    return jsonify({'ok': True})


@app.route('/set_wp1', methods=['POST'])
def set_wp1():
    data = request.json
    lat = float(data['lat'])
    lng = float(data['lng'])

    if ros_node:
        ros_node.pub_clear_waypoints()
        time.sleep(0.05)
        ros_node.pub_waypoint(lat, lng)

    socketio.emit('status', {
        'msg': f'[MANUAL WP1] lat={lat:.7f} lon={lng:.7f}'
    })
    return jsonify({'ok': True, 'lat': lat, 'lng': lng})


@app.route('/cmd/<cmd>', methods=['POST'])
def cmd(cmd):
    if ros_node:
        if cmd == 'takeoff':
            ros_node.pub_takeoff()
        elif cmd == 'start':
            ros_node.pub_start()
        elif cmd == 'land':
            ros_node.pub_land()

    socketio.emit('status', {'msg': f'[CMD] {cmd.upper()} sent'})
    return jsonify({'ok': True})


@app.route('/clear', methods=['POST'])
def clear():
    if ros_node:
        ros_node.pub_clear_waypoints()
    socketio.emit('status', {'msg': 'Waypoints cleared'})
    return jsonify({'ok': True})


def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = WebMapNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    print("=" * 50)
    print("UAV Mission Monitor V3 running at: http://0.0.0.0:5000")
    print("=" * 50)

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
