// ============================================================
// PRAYAGRAJ FLOOD DETECTION - DAY 2 DATA ACQUISITION SCRIPT
// Paste this into code.earthengine.google.com and click "Run"
// ============================================================

// ---- 1. DEFINE STUDY AREA ----
// Bounding box converted from: (25,31,21.55N, 81,39,17.3E) to (25,17,22.08N, 81,57,39.17E)
var aoi = ee.Geometry.Rectangle([81.6548, 25.2895, 81.9609, 25.5227]);

Map.centerObject(aoi, 11);
Map.addLayer(aoi, {color: 'red'}, 'Study Area (AOI)');

// ---- 2. DEFINE DATE WINDOWS ----
// Using the Sept 2019 Prayagraj flood (Ganga/Yamuna crossed danger mark ~Sept 17-18,
// peaked ~Sept 21-22, began receding after Sept 22).
// EDIT THESE if you switch to the Aug 2021 event or adjust for cloud cover.

var preFloodStart  = '2019-07-01';
var preFloodEnd    = '2019-08-31';

var floodStart      = '2019-09-10';
var floodEnd        = '2019-09-30';

var postFloodStart  = '2019-10-05';
var postFloodEnd    = '2019-10-20';

// ---- 3. SENTINEL-2 (optical) - PRE-FLOOD BASELINE ----
var s2PreCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(preFloodStart, preFloodEnd)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
  .sort('CLOUDY_PIXEL_PERCENTAGE');

print('Number of pre-flood S2 candidates found:', s2PreCollection.size());

var s2Pre = s2PreCollection.first().clip(aoi);

print('Pre-flood Sentinel-2 image info:', s2Pre);

Map.addLayer(s2Pre, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Pre-flood S2 (True Color)');

// ---- 4. SENTINEL-2 (optical) - POST-FLOOD ----
var s2Post = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(postFloodStart, postFloodEnd)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first()
  .clip(aoi);

print('Post-flood Sentinel-2 image info:', s2Post);

Map.addLayer(s2Post, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Post-flood S2 (True Color)');

// ---- 5. SENTINEL-1 SAR - DURING FLOOD (sees through clouds!) ----
var s1FloodCollection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate(floodStart, floodEnd)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

print('Number of flood-window S1 SAR candidates found:', s1FloodCollection.size());
print('All flood-window S1 SAR image dates:', s1FloodCollection.aggregate_array('system:time_start'));

var s1Flood = s1FloodCollection.first().clip(aoi);

print('Flood-date Sentinel-1 SAR image info:', s1Flood);

Map.addLayer(s1Flood, {bands: ['VH'], min: -25, max: 0}, 'Flood-date S1 SAR (VH)');

// ---- 6. OPTIONAL: Sentinel-1 SAR pre-flood (for SAR-based change detection) ----
var s1Pre = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate(preFloodStart, preFloodEnd)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .first()
  .clip(aoi);

print('Pre-flood Sentinel-1 SAR image info:', s1Pre);
Map.addLayer(s1Pre, {bands: ['VH'], min: -25, max: 0}, 'Pre-flood S1 SAR (VH)');

// ---- 7. EXPORT ALL IMAGES TO GOOGLE DRIVE ----
// After running, go to the "Tasks" tab (top right) and click "Run" on each export.

Export.image.toDrive({
  image: s2Pre.select(['B2','B3','B4','B8','B11']), // Blue, Green, Red, NIR, SWIR1
  description: 'prayagraj_preflood_S2',
  folder: 'flood_project',
  fileNamePrefix: 'prayagraj_preflood_S2',
  region: aoi,
  scale: 10,
  maxPixels: 1e9
});

Export.image.toDrive({
  image: s2Post.select(['B2','B3','B4','B8','B11']),
  description: 'prayagraj_postflood_S2',
  folder: 'flood_project',
  fileNamePrefix: 'prayagraj_postflood_S2',
  region: aoi,
  scale: 10,
  maxPixels: 1e9
});

Export.image.toDrive({
  image: s1Flood.select(['VH','VV']),
  description: 'prayagraj_flooddate_S1_SAR',
  folder: 'flood_project',
  fileNamePrefix: 'prayagraj_flooddate_S1_SAR',
  region: aoi,
  scale: 10,
  maxPixels: 1e9
});

Export.image.toDrive({
  image: s1Pre.select(['VH','VV']),
  description: 'prayagraj_preflood_S1_SAR',
  folder: 'flood_project',
  fileNamePrefix: 'prayagraj_preflood_S1_SAR',
  region: aoi,
  scale: 10,
  maxPixels: 1e9
});

// ============================================================
// NOTES:
// - If s2Pre or s2Post print as "null" in the console, it means
//   no cloud-free image was found in that window. Widen the
//   date range (e.g. preFloodStart/preFloodEnd) and re-run.
// - If s1Flood is null, double check the date window overlaps
//   an actual Sentinel-1 pass over this region (revisit is ~6-12 days).
// - After exports finish (check the "Tasks" tab), download the
//   GeoTIFFs from your Google Drive "flood_project" folder.
// ============================================================
