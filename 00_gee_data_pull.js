// ============================================================
// PRAYAGRAJ FLOOD DETECTION - MULTI-EVENT & PRECURSORS
// Paste this into code.earthengine.google.com and click "Run"
// ============================================================

// ---- 1. DEFINE STUDY AREA ----
var aoi = ee.Geometry.Rectangle([81.6548, 25.2895, 81.9609, 25.5227]);

Map.centerObject(aoi, 11);
Map.addLayer(aoi, {color: 'red'}, 'Study Area (AOI)');

// ---- 2. STATIC FEATURES (Export Once) ----

// A. Topographical Data (SRTM DEM)
var srtm = ee.Image('USGS/SRTMGL1_003').clip(aoi);
var elevation = srtm.select('elevation');
var slope = ee.Terrain.slope(elevation);
var topoFeatures = elevation.addBands(slope).rename(['elevation', 'slope']);

Export.image.toDrive({
  image: topoFeatures,
  description: 'prayagraj_topo_features',
  folder: 'flood_project',
  fileNamePrefix: 'prayagraj_topo_features',
  region: aoi,
  scale: 30, // SRTM resolution
  maxPixels: 1e9
});

// B. Hydrological Context (Distance to River via JRC Global Surface Water)
var jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi);
// Water occurrence > 80% represents permanent rivers (Ganga/Yamuna)
var permanentWater = jrc.select('occurrence').gt(80);
var riverMask = permanentWater.updateMask(permanentWater);
var distanceToRiver = riverMask.distance(ee.Kernel.euclidean(5000, 'meters')).rename('distance_to_river');

Export.image.toDrive({
  image: distanceToRiver,
  description: 'prayagraj_distance_to_river',
  folder: 'flood_project',
  fileNamePrefix: 'prayagraj_distance_to_river',
  region: aoi,
  scale: 30,
  maxPixels: 1e9
});


// ---- 3. DEFINE FLOOD EVENTS ----
var events = [
  {
    year: '2019',
    preFloodStart: '2019-07-01', preFloodEnd: '2019-08-31',
    floodStart: '2019-09-10', floodEnd: '2019-09-30',
    postFloodStart: '2019-10-05', postFloodEnd: '2019-10-20'
  },
  {
    year: '2021',
    preFloodStart: '2021-06-01', preFloodEnd: '2021-07-31',
    floodStart: '2021-08-10', floodEnd: '2021-08-20',
    postFloodStart: '2021-09-01', postFloodEnd: '2021-09-20'
  },
  {
    year: '2026',
    preFloodStart: '2026-06-01', preFloodEnd: '2026-07-31',
    floodStart: '2026-08-20', floodEnd: '2026-08-31',
    postFloodStart: '2026-09-05', postFloodEnd: '2026-09-20'
  }
];

// ---- 4. PROCESS EACH EVENT ----
events.forEach(function(event) {
  
  // A. OPTICAL CONTEXT: PRE-FLOOD S2 (For NDWI)
  var s2Pre = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.preFloodStart, event.preFloodEnd)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first();
    
  if (s2Pre) {
    Export.image.toDrive({
      image: s2Pre.clip(aoi).select(['B2','B3','B4','B8','B11']),
      description: 'prayagraj_preflood_S2_' + event.year,
      folder: 'flood_project',
      fileNamePrefix: 'prayagraj_preflood_S2_' + event.year,
      region: aoi,
      scale: 10,
      maxPixels: 1e9
    });
  }

  // B. TARGET VARIABLE: DURING FLOOD S1 SAR
  var s1Flood = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(event.floodStart, event.floodEnd)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .first();
    
  if (s1Flood) {
    Export.image.toDrive({
      image: s1Flood.clip(aoi).select(['VH','VV']),
      description: 'prayagraj_flooddate_S1_SAR_' + event.year,
      folder: 'flood_project',
      fileNamePrefix: 'prayagraj_flooddate_S1_SAR_' + event.year,
      region: aoi,
      scale: 10,
      maxPixels: 1e9
    });
  }

  // C. TARGET VARIABLE: POST-FLOOD S2 (For creating ground truth masks)
  var s2Post = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.postFloodStart, event.postFloodEnd)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first();
    
  if (s2Post) {
    Export.image.toDrive({
      image: s2Post.clip(aoi).select(['B2','B3','B4','B8','B11']),
      description: 'prayagraj_postflood_S2_' + event.year,
      folder: 'flood_project',
      fileNamePrefix: 'prayagraj_postflood_S2_' + event.year,
      region: aoi,
      scale: 10,
      maxPixels: 1e9
    });
  }

  // D. METEOROLOGICAL CONTEXT: PRECIPITATION (7, 10, and 30 days prior to floodStart)
  var floodStartDate = ee.Date(event.floodStart);
  
  // 10-Day Rainfall (Legacy, keeping just in case)
  var precip10 = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(aoi)
    .filterDate(floodStartDate.advance(-10, 'day'), floodStartDate)
    .sum().clip(aoi).rename('precipitation');
    
  Export.image.toDrive({
    image: precip10, description: 'prayagraj_precip_10d_' + event.year,
    folder: 'flood_project', fileNamePrefix: 'prayagraj_precip_10d_' + event.year,
    region: aoi, scale: 5566, maxPixels: 1e9
  });

  // 7-Day Rainfall
  var precip7 = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(aoi)
    .filterDate(floodStartDate.advance(-7, 'day'), floodStartDate)
    .sum().clip(aoi).rename('precipitation');
    
  Export.image.toDrive({
    image: precip7, description: 'prayagraj_precip_7d_' + event.year,
    folder: 'flood_project', fileNamePrefix: 'prayagraj_precip_7d_' + event.year,
    region: aoi, scale: 5566, maxPixels: 1e9
  });

  // 30-Day Rainfall
  var precip30 = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(aoi)
    .filterDate(floodStartDate.advance(-30, 'day'), floodStartDate)
    .sum().clip(aoi).rename('precipitation');
    
  Export.image.toDrive({
    image: precip30, description: 'prayagraj_precip_30d_' + event.year,
    folder: 'flood_project', fileNamePrefix: 'prayagraj_precip_30d_' + event.year,
    region: aoi, scale: 5566, maxPixels: 1e9
  });

});

print("Exports queued! Go to the Tasks tab to run them.");
