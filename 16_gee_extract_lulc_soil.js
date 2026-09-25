// ============================================================
// GEE Script: Extract LULC + Soil Clay for Prayagraj
// Paste this into https://code.earthengine.google.com/
// ============================================================

// 1. Define study area (same as your other exports)
var prayagraj = ee.Geometry.Rectangle([81.65, 25.30, 82.05, 25.55]);

// ============================================================
// LAYER 1: ESA WorldCover 2021 (10m Land Use / Land Cover)
// Classes: 10=Tree, 20=Shrub, 30=Grassland, 40=Cropland,
//          50=Built-up, 60=Bare, 80=Water, 90=Wetland, 95=Mangrove
// ============================================================
var worldcover = ee.ImageCollection("ESA/WorldCover/v200")
    .first()
    .clip(prayagraj);

Export.image.toDrive({
    image: worldcover,
    description: 'prayagraj_LULC_ESA_WorldCover',
    folder: 'EarthEngine_Exports',
    region: prayagraj,
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e10
});

// ============================================================
// LAYER 2: Soil Clay Content (0-5cm depth) from OpenLandMap
// Values in % (0-100). Higher clay = worse drainage = more flood
// ============================================================
var soilClay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
    .select('b0')  // 0-5cm depth
    .clip(prayagraj);

Export.image.toDrive({
    image: soilClay,
    description: 'prayagraj_soil_clay_pct',
    folder: 'EarthEngine_Exports',
    region: prayagraj,
    scale: 250,   // Native resolution is 250m
    crs: 'EPSG:4326',
    maxPixels: 1e10
});

print('LULC Preview:', worldcover);
print('Soil Clay Preview:', soilClay);

// Click RUN, then go to the Tasks tab and click RUN on both exports.
// Download the two .tif files and place them in data/ folder:
//   data/prayagraj_LULC_ESA_WorldCover.tif
//   data/prayagraj_soil_clay_pct.tif
