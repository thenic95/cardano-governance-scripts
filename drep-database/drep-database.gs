/***** CONFIG *****/
const BLOCKFROST_BASE = 'https://cardano-mainnet.blockfrost.io/api/v0';
const BLOCKFROST_PROJECT_ID = 'mainnetDsWo03DeberqhCRoVPIrXmczSGxbwLET';
const DREPS_SHEET_NAME = 'DReps';
const MANUAL_DATA_SHEET_NAME = 'Manual Data';
const LOG_SHEET_NAME = 'Execution Log'; // New sheet for the audit log
const STATUS_RANGE_A1 = 'A1:F4';
const TABLE_START_ROW = 5;
const RUN_EVERY_HOURS = 12;
/******************/

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('DRep Tools')
    .addItem('Refresh DReps now (Full)', 'refreshDRepsNow')
    .addItem('Update with Manual Data Only', 'mergeManualDataNow')
    .addSeparator()
    .addItem('Install 12h auto-refresh', 'installTriggerEvery12h')
    .addItem('Remove all triggers', 'deleteAllTriggers_')
    .addToUi();
}

function installTriggerEvery12h() {
  deleteAllTriggers_();
  ScriptApp.newTrigger('refreshDRepsNow').timeBased().everyHours(RUN_EVERY_HOURS).create();
  toast_('Installed 12h trigger.');
}

function deleteAllTriggers_() {
  var triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    ScriptApp.deleteTrigger(triggers[i]);
  }
  toast_('Deleted all triggers.');
}

// Main orchestrator for full data refresh
function refreshDRepsNow() {
  var startTime = new Date();
  var processedCount = 0;
  try {
    var ss = SpreadsheetApp.getActive();
    var drepsSheet = ss.getSheetByName(DREPS_SHEET_NAME);
    if (!drepsSheet) drepsSheet = ss.insertSheet(DREPS_SHEET_NAME);
    drepsSheet.clear();

    initStatus_(drepsSheet);
    updateStatus_(drepsSheet, 'start', { step: 'Discovering DReps via Blockfrost...' });

    var allDRepIDs = fetchAllDRepIDs_Blockfrost_();
    if (!allDRepIDs || allDRepIDs.length === 0) {
      throw new Error('Failed to fetch DRep list from Blockfrost.');
    }
    
    var dRepIDsToProcess = allDRepIDs.slice(0, 100);
    processedCount = dRepIDsToProcess.length;

    updateStatus_(drepsSheet, 'info', { step: 'Found ' + allDRepIDs.length + ' DReps. Processing all of them...' });
    var apiRows = buildOutputRows_(dRepIDsToProcess);

    updateStatus_(drepsSheet, 'info', { step: 'Merging with manual data...' });
    var finalRows = mergeWithManualData_(apiRows);

    updateStatus_(drepsSheet, 'info', { step: 'Writing ' + (finalRows.length - 1) + ' rows...' });
    writeTable_(drepsSheet, finalRows);
    updateSummaryStats_(drepsSheet, finalRows);

    updateStatus_(drepsSheet, 'done', { total: finalRows.length - 1 });
    autoResize_(drepsSheet);
    toast_('DReps refreshed: ' + (finalRows.length - 1));
    logExecution_(startTime, 'Success', processedCount, 'Full Refresh');
  } catch (e) {
    logExecution_(startTime, 'Failure', processedCount, 'Full Refresh: ' + e.message);
    toast_('Script failed: ' + e.message);
  }
}

function mergeManualDataNow() {
    var startTime = new Date();
    var updatedCount = 0;
    try {
        var ss = SpreadsheetApp.getActive();
        var drepsSheet = ss.getSheetByName(DREPS_SHEET_NAME);
        var manualSheet = ss.getSheetByName(MANUAL_DATA_SHEET_NAME);

        if (!drepsSheet || !manualSheet) {
            throw new Error('Both "DReps" and "Manual Data" sheets must exist.');
        }

        toast_('Starting manual data merge...');
        
        var dataRange = drepsSheet.getRange(TABLE_START_ROW, 1, drepsSheet.getLastRow() - TABLE_START_ROW + 1, drepsSheet.getLastColumn());
        var drepsData = dataRange.getValues();
        var manualData = manualSheet.getDataRange().getValues();

        if (drepsData.length < 1) throw new Error('"DReps" sheet has no data to merge.');
        if (manualData.length < 2) {
            toast_('No manual data to merge.');
            logExecution_(startTime, 'Success', 0, 'Manual Merge Only - No data found');
            return;
        }

        var drepsHeaders = drepsData.shift(); 
        var manualHeaders = manualData.shift();
        
        var drepsHeaderMap = {};
        drepsHeaders.forEach(function(header, i) { drepsHeaderMap[header] = i; });

        var drepIdDrepIndex = drepsHeaderMap['DRep ID'];
        if (drepIdDrepIndex === undefined) throw new Error('"DRep ID" column not found in DReps sheet on row ' + TABLE_START_ROW);
        if (manualHeaders[0] !== 'DRep ID') throw new Error('First column in "Manual Data" sheet must be "DRep ID".');

        var manualDataMap = new Map();
        manualData.forEach(function(row) {
            var drepId = row[0];
            if (drepId) {
                var rowObject = {};
                manualHeaders.forEach(function(header, i) {
                    rowObject[header] = row[i];
                });
                manualDataMap.set(drepId, rowObject);
            }
        });

        for (var i = 0; i < drepsData.length; i++) {
            var drepRow = drepsData[i];
            var drepId = drepRow[drepIdDrepIndex];
            
            if (manualDataMap.has(drepId)) {
                var manualRowObject = manualDataMap.get(drepId);
                var hasUpdate = false;

                for (var header in manualRowObject) {
                    var manualValue = manualRowObject[header];
                    if (manualValue !== '' && drepsHeaderMap.hasOwnProperty(header)) {
                        var targetColIndex = drepsHeaderMap[header];
                        drepRow[targetColIndex] = manualValue;
                        hasUpdate = true;
                    }
                }
                if (hasUpdate) updatedCount++;
            }
        }
        
        drepsData.unshift(drepsHeaders);
        dataRange.setValues(drepsData);
        
        updateSummaryStats_(drepsSheet, drepsData);
        autoResize_(drepsSheet);
        
        toast_('Successfully merged manual data. Updated ' + updatedCount + ' DReps.');
        logExecution_(startTime, 'Success', updatedCount, 'Manual Merge Only');
    } catch (e) {
        logExecution_(startTime, 'Failure', updatedCount, 'Manual Merge: ' + e.message);
        toast_('Merge failed: ' + e.message);
    }
}


/* ================================
   Fetchers (Blockfrost)
   ================================ */

function fetchAllDRepIDs_Blockfrost_() {
  var allDRepIDs = [];
  var page = 1;
  var count = 100;
  
  var excludedIDs = ['drep_always_abstain', 'drep_always_no_confidence'];

  while (true) {
    try {
      var url = BLOCKFROST_BASE + '/governance/dreps?page=' + page + '&count=' + count;
      var res = urlFetch_({ url: url, method: 'get' });
      var arr = parseJsonSafe_(res.getContentText());

      if (!Array.isArray(arr) || arr.length === 0) break; 
      
      for (var i = 0; i < arr.length; i++) {
        var drepId = arr[i] ? arr[i].drep_id : null;
        if (drepId && excludedIDs.indexOf(drepId) === -1) {
          allDRepIDs.push(drepId);
        }
      }
      
      if (arr.length < count) break;
      
      page++;
      Utilities.sleep(50);
    } catch (e) {
      console.error('Error fetching DRep list page ' + page + ': ' + e.message);
      break;
    }
  }
  return allDRepIDs;
}

/* ================================
   Build & Merge
   ================================ */

function buildOutputRows_(allDRepIDs) {
  var headers = [
    'Source','DRep ID','Active','Retired','Expired','Name','Email','X/Twitter','LinkedIn','GitHub', 'Website', 'Other Links (Count)','Payment Address','Image URL',
    'Voting Power (lovelace)','Voting Power (ADA)',
    'Metadata URL','Anchor Hash','Last Updated (UTC)'
  ];
  var rows = [headers];

  for (var i = 0; i < allDRepIDs.length; i++) {
    var drep_id = allDRepIDs[i];
    var drepDetails = null;
    var drepMetadata = null;

    // Fetch details with retry; skip DRep only if details ultimately fail
    try {
      var detailsUrl = BLOCKFROST_BASE + '/governance/dreps/' + drep_id;
      var detailsRes = urlFetchWithRetry_({ url: detailsUrl, method: 'get' }, 3, 400);
      drepDetails = parseJsonSafe_(detailsRes.getContentText());
      Utilities.sleep(60);
    } catch (e) {
      console.error('Failed to fetch details for ' + drep_id + ': ' + e.message);
      continue;
    }

    // Fetch metadata with retry; do not skip the DRep if metadata fails
    if (drepDetails) {
      try {
        var metadataUrl = BLOCKFROST_BASE + '/governance/dreps/' + drep_id + '/metadata';
        var metadataRes = urlFetchWithRetry_({ url: metadataUrl, method: 'get' }, 2, 400);
        drepMetadata = parseJsonSafe_(metadataRes.getContentText());
      } catch (e) {
        console.warn('Metadata fetch failed for ' + drep_id + ': ' + e.message);
        drepMetadata = null;
      }
      Utilities.sleep(60);
    }

    if (!drepDetails || typeof drepDetails !== 'object') {
        console.error('Skipping DRep due to invalid details response for ID: ' + drep_id);
        continue;
    }

    var vp = firstNumber_([drepDetails.amount]);
    var metaUrl = '', metaHash = '', name = '', paymentAddr = '', imageUrl = '', email = '', twitterUrl = '', linkedinUrl = '', githubUrl = '', websiteUrl = '', otherLinksCount = 0;

    if (drepMetadata) {
      metaUrl = drepMetadata.url || '';
      metaHash = drepMetadata.hash || '';
      var mdBody = drepMetadata.json_metadata ? drepMetadata.json_metadata.body : null;
      
      if (mdBody) {
        name = cleanMetadataField_(mdBody.givenName || mdBody.dRepName || mdBody.name || mdBody.handle || mdBody.alias || mdBody.title);
        paymentAddr = cleanMetadataField_(mdBody.paymentAddress || mdBody.payment_address);
        email = cleanMetadataField_(mdBody.email || mdBody.contact || mdBody.emailAddress);
        
        var rawImage = mdBody.image || mdBody.logoUrl || mdBody.logo;
        imageUrl = (rawImage && typeof rawImage === 'object' && !Array.isArray(rawImage)) ? cleanMetadataField_(rawImage.contentUrl || rawImage.url) : cleanMetadataField_(rawImage);
        
        var linksRaw = mdBody.references || mdBody.dRepWebsite || mdBody.socialMedia || mdBody.links || mdBody.urls || [];
        var normalizedLinks = normalizeLinks_(linksRaw);

        if (!email) {
            for (var j = 0; j < normalizedLinks.length; j++) {
                if (normalizedLinks[j].url && typeof normalizedLinks[j].url === 'string' && normalizedLinks[j].url.startsWith('mailto:')) {
                    email = normalizedLinks[j].url.substring(7);
                    break;
                }
            }
        }
        
        var remainingLinks = [];
        normalizedLinks.forEach(function(link) {
          if (link.url && typeof link.url === 'string') {
            if (!twitterUrl && (link.url.includes('twitter.com') || link.url.includes('x.com'))) {
              twitterUrl = link.url;
            } else if (!linkedinUrl && link.url.includes('linkedin.com')) {
              linkedinUrl = link.url;
            } else if (!githubUrl && link.url.includes('github.com')) {
              githubUrl = link.url;
            } else if (!websiteUrl && (link.label === 'Website' || link.label === 'Homepage' || link.label === 'Link' || link.label === 'dRepWebsite')) {
               websiteUrl = link.url;
            }
            else {
              remainingLinks.push(link);
            }
          }
        });
        otherLinksCount = remainingLinks.length;
      }
    }

    // Ensure boolean fields are actual booleans (handle booleans, 1/0 and strings)
    var retiredBool = toBoolCell_(drepDetails.retired);
    var expiredBool = toBoolCell_(drepDetails.expired);
    var activeBool = null;
    if (drepDetails.hasOwnProperty('active')) {
      activeBool = toBoolCell_(drepDetails.active);
    }
    // If active is missing or contradictory to retired/expired, derive it
    var derivedActive = (!retiredBool && !expiredBool);
    if (activeBool === null) {
      activeBool = derivedActive;
    } else if (activeBool === false && derivedActive === true) {
      activeBool = true;
    }

    var rowData = [
      'Blockfrost',
      drepDetails.drep_id,
      activeBool,
      retiredBool,
      expiredBool,
      name, email, twitterUrl, linkedinUrl, githubUrl, websiteUrl, otherLinksCount, paymentAddr, imageUrl,
      isFinite(vp) ? vp : 0, isFinite(vp) ? toAda_(vp) : 0, metaUrl, metaHash, isoNow_()
    ];
    
    rows.push(rowData.map(function(cell) { return truncateValue_(cell); }));
  }
  return rows;
}

function mergeWithManualData_(apiRows) {
  var ss = SpreadsheetApp.getActive();
  var manualSheet = ss.getSheetByName(MANUAL_DATA_SHEET_NAME);

  if (!manualSheet) {
    manualSheet = ss.insertSheet(MANUAL_DATA_SHEET_NAME);
    var manualHeaders = ['DRep ID', 'Email', 'Affiliation', 'My Custom Notes'];
    manualSheet.getRange(1, 1, 1, manualHeaders.length).setValues([manualHeaders]).setFontWeight('bold');
    manualSheet.setFrozenRows(1);
    toast_('Created "Manual Data" sheet. Add your custom info there.');
    return apiRows;
  }

  var manualData = manualSheet.getDataRange().getValues();
  if (manualData.length < 2) return apiRows;

  var manualHeaders = manualData.shift();
  if (manualHeaders[0] !== 'DRep ID') {
    toast_('ERROR: First column in "Manual Data" sheet must be "DRep ID".');
    return apiRows;
  }

  var manualDataMap = new Map();
  for (var i = 0; i < manualData.length; i++) {
    var drepId = manualData[i][0];
    if (drepId) {
      var rowObject = {};
      for (var j = 0; j < manualHeaders.length; j++) {
        rowObject[manualHeaders[j]] = manualData[i][j];
      }
      manualDataMap.set(drepId, rowObject);
    }
  }

  var apiHeaders = apiRows.shift();
  
  var finalHeaders = [].concat(apiHeaders);
  var manualHeadersToAdd = manualHeaders.slice(1).filter(function(h) {
      return apiHeaders.indexOf(h) === -1;
  });
  finalHeaders = finalHeaders.concat(manualHeadersToAdd);

  var drepIdApiIndex = apiHeaders.indexOf('DRep ID');
  var finalRows = [finalHeaders];

  for (var k = 0; k < apiRows.length; k++) {
    var apiRow = apiRows[k];
    var drepId = apiRow[drepIdApiIndex];
    
    var finalRowObject = {};
    for (var l = 0; l < apiHeaders.length; l++) {
      finalRowObject[apiHeaders[l]] = apiRow[l];
    }
    
    var manualRowObject = manualDataMap.get(drepId);

    if (manualRowObject) {
      // Do not allow manual data to override computed/system columns
      var lockedCols = {
        'Active': true,
        'Retired': true,
        'Expired': true,
        'Voting Power (lovelace)': true,
        'Voting Power (ADA)': true,
        'Metadata URL': true,
        'Anchor Hash': true,
        'Last Updated (UTC)': true,
        'Source': true,
        'DRep ID': true,
        'Payment Address': true,
        'Image URL': true
      };
      for (var header in manualRowObject) {
        if (manualRowObject[header] !== '' && !lockedCols[header]) {
          finalRowObject[header] = manualRowObject[header];
        }
      }
    }
    
    var finalRowArray = [];
    for (var m = 0; m < finalHeaders.length; m++) {
      var cellVal = finalRowObject[finalHeaders[m]];
      // Preserve false/0 values; only replace truly undefined with empty string
      finalRowArray.push(cellVal === undefined ? '' : cellVal);
    }
    finalRows.push(finalRowArray);
  }
  
  // Coerce boolean columns to true/false primitives for consistency
  var activeIdx = finalHeaders.indexOf('Active');
  var retiredIdx = finalHeaders.indexOf('Retired');
  var expiredIdx = finalHeaders.indexOf('Expired');
  if (activeIdx !== -1 || retiredIdx !== -1 || expiredIdx !== -1) {
    for (var r = 1; r < finalRows.length; r++) {
      if (activeIdx !== -1) finalRows[r][activeIdx] = toBoolCell_(finalRows[r][activeIdx]);
      if (retiredIdx !== -1) finalRows[r][retiredIdx] = toBoolCell_(finalRows[r][retiredIdx]);
      if (expiredIdx !== -1) finalRows[r][expiredIdx] = toBoolCell_(finalRows[r][expiredIdx]);
    }
  }
  
  return finalRows;
}

/* ================================
   Write, Log & Helpers
   ================================ */

function logExecution_(startTime, status, processedCount, notes) {
    var ss = SpreadsheetApp.getActive();
    var logSheet = ss.getSheetByName(LOG_SHEET_NAME);
    if (!logSheet) {
        logSheet = ss.insertSheet(LOG_SHEET_NAME, 0); 
        var headers = ['Run Start Time (UTC)', 'Run End Time (UTC)', 'Duration (sec)', 'Status', 'DReps Processed', 'Notes / Error'];
        logSheet.getRange(1, 1, 1, headers.length).setValues([headers]).setFontWeight('bold');
        logSheet.setFrozenRows(1);
    }
    
    var endTime = new Date();
    var duration = (endTime.getTime() - startTime.getTime()) / 1000;
    
    var logData = [
        startTime.toISOString(),
        endTime.toISOString(),
        duration.toFixed(2),
        status,
        processedCount,
        notes
    ];
    
    logSheet.insertRowAfter(1);
    logSheet.getRange(2, 1, 1, logData.length).setValues([logData]);
}


function writeTable_(sh, rows) {
  if (rows.length > 0) {
    sh.getRange(TABLE_START_ROW, 1, rows.length, rows[0].length).setValues(rows);
    sh.getRange(TABLE_START_ROW, 1, 1, rows[0].length).setFontWeight('bold');
  }
}

function writeEmptyTable_(sh) {
  var headers = [
    'Source','DRep ID','Active','Retired','Expired','Name','Email','X/Twitter','LinkedIn','GitHub', 'Website', 'Other Links (Count)','Payment Address','Image URL',
    'Voting Power (lovelace)','Voting Power (ADA)',
    'Metadata URL','Anchor Hash','Last Updated (UTC)'
  ];
  sh.getRange(TABLE_START_ROW, 1, 1, headers.length).setValues([headers]).setFontWeight('bold');
}

function truncateValue_(value) {
    const LIMIT = 2500;
    if (typeof value === 'string' && value.length > LIMIT) {
        return value.substring(0, LIMIT - 15) + '...[TRUNCATED]';
    }
    return value;
}

function cleanMetadataField_(field) {
  if (!field) return '';
  if (typeof field !== 'object') return String(field);
  if (Array.isArray(field)) return cleanMetadataField_(field[0]);
  if (field['@value']) return String(field['@value']);
  return '';
}

function urlFetch_(opts) {
  var url = opts.url;
  var method = (opts.method || 'get').toUpperCase();
  var headers = { 'project_id': BLOCKFROST_PROJECT_ID };
  var fetchOpts = { method: method, headers: headers, muteHttpExceptions: true, followRedirects: true };
  var res = UrlFetchApp.fetch(url, fetchOpts);
  var code = res.getResponseCode();
  if (code < 200 || code >= 300) {
    if (code === 404 && (opts.url.includes('/metadata') || opts.url.includes('/governance/dreps/'))) {
      return { getContentText: function() { return 'null'; } };
    }
    var msg = method + ' ' + url + ' -> HTTP ' + code + ': ' + res.getContentText().slice(0, 300);
    throw new Error(msg);
  }
  return res;
}

// Wrapper to retry transient failures (network/5xx/Address unavailable/etc.) with backoff
function urlFetchWithRetry_(opts, retries, baseDelayMs) {
  var attempts = Math.max(1, retries || 3);
  var delay = Math.max(50, baseDelayMs || 400);
  for (var i = 1; i <= attempts; i++) {
    try {
      return urlFetch_(opts);
    } catch (e) {
      var msg = e && e.message ? e.message : String(e);
      // If last attempt, rethrow
      if (i === attempts) throw e;
      // Backoff then retry
      Utilities.sleep(delay);
      delay = Math.min(delay * 2, 8000);
    }
  }
}

function parseJsonSafe_(txt) {
  try { return JSON.parse(txt); } catch (e) { return null; }
}

function firstNumber_(arr) {
  for (var i = 0; i < arr.length; i++) {
    var n = normalizeNumber_(arr[i]);
    if (isFinite(n)) return n;
  }
  return NaN;
}

function normalizeNumber_(v) {
  if (v == null) return NaN;
  if (typeof v === 'number') return v;
  if (typeof v === 'string') {
    var s = v.replace(/,/g, '').trim();
    if (!s) return NaN;
    return Number.isFinite(Number(s)) ? Number(s) : NaN;
  }
  return NaN;
}

function toAda_(lovelace) {
  return Number(lovelace) / 1e6;
}

function normalizeLinks_(links) {
  if (!links) return [];
  if (typeof links === 'string') return [{ label: 'Link', url: links }];
  if (!(links instanceof Array)) {
     if (typeof links === 'object') {
        return Object.keys(links).map(function(key) {
            return { label: key, url: links[key] };
        });
    }
    return [];
  }
  var out = [];
  for (var i = 0; i < links.length; i++) {
    var x = links[i];
    if (typeof x === 'string') out.push({ label: 'Link', url: x });
    else if (x && typeof x === 'object') out.push({ label: x.label || x.name || 'Link', url: x.uri || x.url || x.href || '' });
  }
  return out.filter(function(l) { return !!l.url; });
}

function stringifyLinks_(linksArr) {
  if (!(linksArr instanceof Array) || !linksArr.length) return '';
  var parts = linksArr.map(function(l) {
    return l.label ? (l.label + ': ' + l.url) : l.url;
  });
  return parts.join(' | ');
}

// Normalize a variety of truthy/falsey inputs to boolean primitives for Sheets
function toBoolCell_(v) {
  if (v === true) return true;
  if (v === false) return false;
  if (v == null) return false;
  if (typeof v === 'number') return v !== 0;
  if (typeof v === 'string') {
    var s = v.trim().toLowerCase();
    if (s === 'true' || s === 't' || s === 'yes' || s === 'y' || s === '1') return true;
    if (s === 'false' || s === 'f' || s === 'no' || s === 'n' || s === '0') return false;
    return false;
  }
  return false;
}

function isoNow_() { return new Date().toISOString(); }

function autoResize_(sh) {
  sh.getRange(TABLE_START_ROW, 1, sh.getLastRow(), sh.getLastColumn()).getValues();
  for (var c = 1; c <= sh.getLastColumn(); c++) {
      sh.autoResizeColumn(c);
  }
}

/* ===== Status and Summary ===== */
function initStatus_(sh) {
  sh.getRange(STATUS_RANGE_A1).clearContent().setBackground('#FFFFFF');
  sh.getRange('A1').setValue('DReps status').setFontWeight('bold');
}

function updateStatus_(sh, phase, ctx) {
  var now = isoNow_();
  var msg = '';
  if (phase === 'start') msg = 'Starting... ' + now;
  else if (phase === 'info') msg = (ctx && ctx.step) ? ctx.step : '';
  else if (phase === 'done') msg = 'Done. Processed ' + (ctx && ctx.total ? ctx.total : 0) + ' DReps @ ' + now;
  else if (phase === 'error') msg = 'ERROR: ' + (ctx && ctx.step ? ctx.step : 'An unknown error occurred.');
  sh.getRange('B1').setValue(msg);
}

function updateSummaryStats_(sh, rows) {
  if (rows.length < 2) return;
  var headers = rows[0];
  var activeIndex = headers.indexOf('Active');
  var retiredIndex = headers.indexOf('Retired');
  var expiredIndex = headers.indexOf('Expired');
  var nameIndex = headers.indexOf('Name');
  var emailIndex = headers.indexOf('Email');
  var twitterIndex = headers.indexOf('X/Twitter');
  var linkedinIndex = headers.indexOf('LinkedIn');
  var githubIndex = headers.indexOf('GitHub');
  var websiteIndex = headers.indexOf('Website');

  sh.getRange('A2').setValue('Provider');
  sh.getRange('B2').setValue('Blockfrost');
  sh.getRange('A3').setValue('Total DReps');
  sh.getRange('B3').setValue(rows.length - 1);
  
  sh.getRange('C2').setValue('Active');
  sh.getRange('D2').setValue(rows.slice(1).reduce(function(n, r) {
    var v = r[activeIndex];
    return n + ((v === true) || (typeof v === 'string' && v.toUpperCase() === 'TRUE') ? 1 : 0);
  }, 0));
  sh.getRange('E2').setValue('Retired');
  sh.getRange('F2').setValue(rows.slice(1).reduce(function(n, r) {
    var v = r[retiredIndex];
    return n + ((v === true) || (typeof v === 'string' && v.toUpperCase() === 'TRUE') ? 1 : 0);
  }, 0));
  
  sh.getRange('C3').setValue('Expired');
  sh.getRange('D3').setValue(rows.slice(1).reduce(function(n, r) {
    var v = r[expiredIndex];
    return n + ((v === true) || (typeof v === 'string' && v.toUpperCase() === 'TRUE') ? 1 : 0);
  }, 0));
  sh.getRange('E3').setValue('With name');
  sh.getRange('F3').setValue(rows.slice(1).reduce(function(n, r) { return n + (r[nameIndex] ? 1 : 0); }, 0));

  sh.getRange('C4').setValue('With email');
  sh.getRange('D4').setValue(rows.slice(1).reduce(function(n, r) { return n + (r[emailIndex] ? 1 : 0); }, 0));
  sh.getRange('E4').setValue('With X/Twitter');
  sh.getRange('F4').setValue(rows.slice(1).reduce(function(n, r) { return n + (r[twitterIndex] ? 1 : 0); }, 0));
  
  sh.getRange('A4').setValue('Last run (UTC)');
  sh.getRange('B4').setValue(isoNow_());
  sh.getRange('A2:F4').setBackground('#F7F7F7');
}

function toast_(msg) {
  SpreadsheetApp.getActive().toast(msg, 'DRep Tools', 5);
}
