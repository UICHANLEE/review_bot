#!/usr/bin/env node

const fs = require("node:fs/promises");
const path = require("node:path");

let gplay;
let appStore;
let ExcelJS;

try {
  gplay = require("google-play-scraper");
  gplay = gplay.default || gplay;
  appStore = require("app-store-scraper");
  ExcelJS = require("exceljs");
} catch (error) {
  if (error && error.code === "MODULE_NOT_FOUND") {
    console.error("Missing dependencies. Run `npm install` inside review_bot first.");
    process.exit(1);
  }
  throw error;
}

const STORE_GOOGLE = "google";
const STORE_APPLE = "apple";
const STORE_BOTH = "both";

function parseArgs(argv) {
  const options = {
    keyword: "",
    country: "kr",
    lang: "ko",
    top: 10,
    reviews: 50,
    store: STORE_BOTH,
    out: ""
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];

    if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else if (arg === "--keyword" || arg === "-k") {
      options.keyword = requireValue(arg, next);
      index += 1;
    } else if (arg === "--country") {
      options.country = requireValue(arg, next).toLowerCase();
      index += 1;
    } else if (arg === "--lang") {
      options.lang = requireValue(arg, next).toLowerCase();
      index += 1;
    } else if (arg === "--top") {
      options.top = parsePositiveInt(arg, requireValue(arg, next));
      index += 1;
    } else if (arg === "--reviews") {
      options.reviews = parsePositiveInt(arg, requireValue(arg, next));
      index += 1;
    } else if (arg === "--store") {
      options.store = requireValue(arg, next).toLowerCase();
      index += 1;
    } else if (arg === "--out") {
      options.out = requireValue(arg, next);
      index += 1;
    } else {
      throw new Error(`Unknown option: ${arg}`);
    }
  }

  if (!options.help && !options.keyword.trim()) {
    throw new Error("Required option missing: --keyword");
  }

  if (![STORE_GOOGLE, STORE_APPLE, STORE_BOTH].includes(options.store)) {
    throw new Error("--store must be one of: google, apple, both");
  }

  return options;
}

function requireValue(option, value) {
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${option}`);
  }
  return value;
}

function parsePositiveInt(option, value) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${option} must be a positive integer`);
  }
  return parsed;
}

function printHelp() {
  console.log(`Usage:
  npm start -- --keyword "budget app"

Options:
  -k, --keyword <text>   Search keyword. Required.
      --country <code>   Store country code. Default: kr
      --lang <code>      Google Play language code. Default: ko
      --top <number>     Number of apps per store. Default: 10
      --reviews <number> Number of reviews per app. Default: 50
      --store <name>     google, apple, or both. Default: both
      --out <file>       Output .xlsx path. Default: output/reviews_<keyword>_<timestamp>.xlsx
  -h, --help             Show this help.

Examples:
  npm start -- --keyword "todo" --country us --lang en --reviews 100
  npm start -- --keyword "가계부" --country kr --lang ko --top 10 --reviews 30 --out output/budget_reviews.xlsx`);
}

function buildDefaultOutputPath(keyword) {
  const safeKeyword = keyword
    .trim()
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 40) || "keyword";
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return path.join("output", `reviews_${safeKeyword}_${stamp}.xlsx`);
}

async function collectGoogleApps(options, errors) {
  try {
    const results = await gplay.search({
      term: options.keyword,
      num: options.top,
      country: options.country,
      lang: options.lang,
      fullDetail: false
    });

    return results.slice(0, options.top).map((app, index) => ({
      store: "Google Play",
      rank: index + 1,
      appId: app.appId,
      title: app.title || "",
      developer: app.developer || "",
      url: app.url || `https://play.google.com/store/apps/details?id=${app.appId}`,
      score: app.score ?? "",
      ratings: app.ratings ?? "",
      reviewsCount: app.reviews ?? "",
      price: app.priceText ?? app.price ?? "",
      icon: app.icon || ""
    }));
  } catch (error) {
    errors.push(toErrorRow("Google Play", "search", options.keyword, error));
    return [];
  }
}

async function collectGoogleReviews(app, options, errors) {
  try {
    const payload = await gplay.reviews({
      appId: app.appId,
      sort: gplay.sort && gplay.sort.NEWEST,
      num: options.reviews,
      country: options.country,
      lang: options.lang
    });
    const reviews = Array.isArray(payload) ? payload : payload.data || [];

    return reviews.slice(0, options.reviews).map((review, index) => ({
      store: app.store,
      appRank: app.rank,
      appId: app.appId,
      appTitle: app.title,
      reviewRank: index + 1,
      reviewId: review.id || review.reviewId || "",
      userName: review.userName || review.user || "",
      score: review.score ?? "",
      title: review.title || "",
      text: review.text || review.content || "",
      date: normalizeDate(review.date || review.updated),
      version: review.version || review.appVersion || "",
      thumbsUp: review.thumbsUp ?? review.thumbsUpCount ?? "",
      developerReply: review.replyText || "",
      developerReplyDate: normalizeDate(review.replyDate)
    }));
  } catch (error) {
    errors.push(toErrorRow(app.store, "reviews", app.appId, error));
    return [];
  }
}

async function collectAppleApps(options, errors) {
  try {
    const results = await appStore.search({
      term: options.keyword,
      num: options.top,
      country: options.country
    });

    return results.slice(0, options.top).map((app, index) => ({
      store: "App Store",
      rank: index + 1,
      appId: String(app.id || app.appId || ""),
      title: app.title || "",
      developer: app.developer || app.artistName || "",
      url: app.url || "",
      score: app.score ?? app.averageUserRating ?? "",
      ratings: app.ratings ?? app.userRatingCount ?? "",
      reviewsCount: app.reviews ?? "",
      price: app.price ?? "",
      icon: app.icon || ""
    }));
  } catch (error) {
    errors.push(toErrorRow("App Store", "search", options.keyword, error));
    return [];
  }
}

async function collectAppleReviews(app, options, errors) {
  const reviews = [];
  const pageSizeEstimate = 50;
  const maxPages = Math.min(10, Math.ceil(options.reviews / pageSizeEstimate) + 1);

  for (let page = 1; page <= maxPages && reviews.length < options.reviews; page += 1) {
    try {
      const pageReviews = await appStore.reviews({
        id: app.appId,
        country: options.country,
        sort: appStore.sort && appStore.sort.RECENT,
        page
      });

      if (!Array.isArray(pageReviews) || pageReviews.length === 0) {
        break;
      }

      reviews.push(...pageReviews);
    } catch (error) {
      errors.push(toErrorRow(app.store, "reviews", `${app.appId} page ${page}`, error));
      break;
    }
  }

  return reviews.slice(0, options.reviews).map((review, index) => ({
    store: app.store,
    appRank: app.rank,
    appId: app.appId,
    appTitle: app.title,
    reviewRank: index + 1,
    reviewId: review.id || "",
    userName: review.userName || "",
    score: review.score ?? review.rating ?? "",
    title: review.title || "",
    text: review.text || review.review || "",
    date: normalizeDate(review.updated || review.date),
    version: review.version || "",
    thumbsUp: "",
    developerReply: "",
    developerReplyDate: ""
  }));
}

function normalizeDate(value) {
  if (!value) {
    return "";
  }
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toISOString();
}

function toErrorRow(store, phase, target, error) {
  return {
    store,
    phase,
    target,
    message: error && error.message ? error.message : String(error)
  };
}

async function collectAll(options) {
  const errors = [];
  const apps = [];
  const reviews = [];

  if (options.store === STORE_GOOGLE || options.store === STORE_BOTH) {
    console.log("Searching Google Play...");
    const googleApps = await collectGoogleApps(options, errors);
    apps.push(...googleApps);
    for (const app of googleApps) {
      console.log(`Collecting Google Play reviews: ${app.rank}. ${app.title}`);
      reviews.push(...(await collectGoogleReviews(app, options, errors)));
    }
  }

  if (options.store === STORE_APPLE || options.store === STORE_BOTH) {
    console.log("Searching App Store...");
    const appleApps = await collectAppleApps(options, errors);
    apps.push(...appleApps);
    for (const app of appleApps) {
      console.log(`Collecting App Store reviews: ${app.rank}. ${app.title}`);
      reviews.push(...(await collectAppleReviews(app, options, errors)));
    }
  }

  return { apps, reviews, errors };
}

async function writeWorkbook(filePath, options, data) {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "review_bot";
  workbook.created = new Date();
  workbook.modified = new Date();

  const summary = workbook.addWorksheet("Summary");
  summary.columns = [
    { header: "Field", key: "field", width: 24 },
    { header: "Value", key: "value", width: 60 }
  ];
  summary.addRows([
    { field: "Keyword", value: options.keyword },
    { field: "Country", value: options.country },
    { field: "Language", value: options.lang },
    { field: "Store", value: options.store },
    { field: "Apps per store", value: options.top },
    { field: "Reviews per app", value: options.reviews },
    { field: "Collected apps", value: data.apps.length },
    { field: "Collected reviews", value: data.reviews.length },
    { field: "Errors", value: data.errors.length },
    { field: "Generated at", value: new Date().toISOString() }
  ]);
  styleHeader(summary);

  const appsSheet = workbook.addWorksheet("Apps");
  appsSheet.columns = [
    { header: "Store", key: "store", width: 16 },
    { header: "Rank", key: "rank", width: 10 },
    { header: "App ID", key: "appId", width: 24 },
    { header: "Title", key: "title", width: 34 },
    { header: "Developer", key: "developer", width: 30 },
    { header: "Score", key: "score", width: 10 },
    { header: "Ratings", key: "ratings", width: 12 },
    { header: "Reviews Count", key: "reviewsCount", width: 14 },
    { header: "Price", key: "price", width: 12 },
    { header: "URL", key: "url", width: 60 },
    { header: "Icon", key: "icon", width: 60 }
  ];
  appsSheet.addRows(data.apps);
  styleHeader(appsSheet);

  const reviewsSheet = workbook.addWorksheet("Reviews");
  reviewsSheet.columns = [
    { header: "Store", key: "store", width: 16 },
    { header: "App Rank", key: "appRank", width: 10 },
    { header: "App ID", key: "appId", width: 24 },
    { header: "App Title", key: "appTitle", width: 34 },
    { header: "Review Rank", key: "reviewRank", width: 12 },
    { header: "Review ID", key: "reviewId", width: 26 },
    { header: "User Name", key: "userName", width: 22 },
    { header: "Score", key: "score", width: 10 },
    { header: "Title", key: "title", width: 30 },
    { header: "Text", key: "text", width: 80 },
    { header: "Date", key: "date", width: 26 },
    { header: "Version", key: "version", width: 14 },
    { header: "Thumbs Up", key: "thumbsUp", width: 12 },
    { header: "Developer Reply", key: "developerReply", width: 60 },
    { header: "Developer Reply Date", key: "developerReplyDate", width: 26 }
  ];
  reviewsSheet.addRows(data.reviews);
  styleHeader(reviewsSheet);
  reviewsSheet.getColumn("text").alignment = { wrapText: true, vertical: "top" };
  reviewsSheet.getColumn("developerReply").alignment = { wrapText: true, vertical: "top" };

  if (data.errors.length > 0) {
    const errorsSheet = workbook.addWorksheet("Errors");
    errorsSheet.columns = [
      { header: "Store", key: "store", width: 16 },
      { header: "Phase", key: "phase", width: 14 },
      { header: "Target", key: "target", width: 40 },
      { header: "Message", key: "message", width: 80 }
    ];
    errorsSheet.addRows(data.errors);
    styleHeader(errorsSheet);
  }

  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await workbook.xlsx.writeFile(filePath);
}

function styleHeader(worksheet) {
  worksheet.views = [{ state: "frozen", ySplit: 1 }];
  const header = worksheet.getRow(1);
  header.font = { bold: true };
  header.alignment = { vertical: "middle" };
  worksheet.autoFilter = {
    from: { row: 1, column: 1 },
    to: { row: 1, column: worksheet.columnCount }
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const outputPath = path.resolve(
    process.cwd(),
    options.out || buildDefaultOutputPath(options.keyword)
  );
  const data = await collectAll(options);

  await writeWorkbook(outputPath, options, data);

  console.log("");
  console.log(`Apps: ${data.apps.length}`);
  console.log(`Reviews: ${data.reviews.length}`);
  console.log(`Errors: ${data.errors.length}`);
  console.log(`Excel: ${outputPath}`);
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
