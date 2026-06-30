
import * as webllm from
"https://esm.run/@mlc-ai/web-llm";


import {
    pipeline
}
from
"https://cdn.jsdelivr.net/npm/@xenova/transformers";

const DB_NAME = "rag-db";
const STORE_NAME = "vectors";

let embedder;
let engine;

/*
----------------------------------------
IndexedDB
----------------------------------------
*/

function openDB() {

    return new Promise((resolve, reject) => {

        const request =
            indexedDB.open(DB_NAME, 1);

        request.onupgradeneeded = (e) => {

            const db = e.target.result;

            if (
                !db.objectStoreNames.contains(STORE_NAME)
            ) {
                db.createObjectStore(
                    STORE_NAME,
                    {
                        keyPath: "id"
                    }
                );
            }
        };

        request.onsuccess =
            () => resolve(request.result);

        request.onerror =
            () => reject(request.error);
    });
}

async function saveVector(item) {

    const db = await openDB();

    return new Promise((resolve, reject) => {

        const tx =
            db.transaction(
                STORE_NAME,
                "readwrite"
            );

        tx.objectStore(STORE_NAME)
            .put(item);

        tx.oncomplete =
            () => resolve();

        tx.onerror =
            () => reject(tx.error);
    });
}

async function getAllVectors() {

    const db = await openDB();

    return new Promise((resolve, reject) => {

        const tx =
            db.transaction(
                STORE_NAME,
                "readonly"
            );

        const req =
            tx.objectStore(STORE_NAME)
              .getAll();

        req.onsuccess =
            () => resolve(req.result);

        req.onerror =
            () => reject(req.error);
    });
}

async function clearVectors() {

    const db = await openDB();

    return new Promise((resolve, reject) => {

        const tx =
            db.transaction(
                STORE_NAME,
                "readwrite"
            );

        tx.objectStore(STORE_NAME)
            .clear();

        tx.oncomplete =
            () => resolve();

        tx.onerror =
            () => reject(tx.error);
    });
}

/*
----------------------------------------
Embedding
----------------------------------------
*/

async function initializeEmbedding() {

    embedder =
      await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
      );
}

async function embed(text) {

    const result =
      await embedder(
        text,
        {
          pooling: "mean",
          normalize: true
        }
      );

    return Array.from(result.data);
}

/*
----------------------------------------
Cosine Similarity
----------------------------------------
*/

function cosineSimilarity(a, b) {

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {

        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dot /
        (
            Math.sqrt(normA) *
            Math.sqrt(normB)
        );
}

/*
----------------------------------------
Index Build
----------------------------------------
*/

const INDEX_VERSION = "egov-v4";

async function buildIndex(onProgress) {

    if (
        localStorage.getItem("indexed")
        === INDEX_VERSION
    ) {
        return;
    }

    // 旧バージョンの古いエントリ(title/urlなし)を除去してから再構築
    await clearVectors();

    const docs =
      await fetch("./knowledge.json")
      .then(r => r.json());

    let done = 0;

    for (const doc of docs) {

        const text =
          `${doc.title}\n${doc.content}`;

        // 事前計算済み Embedding があればブラウザ内計算をスキップ
        const embedding =
          Array.isArray(doc.embedding)
            ? doc.embedding
            : await embed(text);

        await saveVector({
            id: doc.id,
            text,
            title: doc.title,
            url: doc.url,
            organization: doc.organization,
            embedding
        });

        done++;

        if (onProgress) {
            onProgress(done, docs.length);
        }
    }

    localStorage.setItem(
        "indexed",
        INDEX_VERSION
    );
}

/*
----------------------------------------
Search
----------------------------------------
*/

async function search(query) {

    const queryEmbedding =
        await embed(query);

    const vectors =
        await getAllVectors();

    return vectors
        .map(v => ({
            ...v,
            score: cosineSimilarity(
                queryEmbedding,
                v.embedding
            )
        }))
        .sort(
            (a,b) =>
                b.score - a.score
        )
        .slice(0, 5);
}

/*
----------------------------------------
WebGPU 判定
----------------------------------------
*/

function hasWebGPU() {
    return typeof navigator !== "undefined"
        && "gpu" in navigator;
}

/*
----------------------------------------
WebLLM
----------------------------------------
*/

async function isWebGPUAvailable() {

    if (!navigator.gpu) {
        return false;
    }

    try {
        const adapter =
          await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch {
        return false;
    }
}

async function initializeLLM() {

    if (!(await isWebGPUAvailable())) {
        engine = null;
        return;
    }

    engine =
      new webllm.MLCEngine();

    await engine.reload(
            "Phi-3.5-mini-instruct-q4f16_1-MLC"
    );
}

/*
----------------------------------------
RAG
----------------------------------------
*/

async function ask(question) {

    const docs =
      await search(question);

    // WebGPU 非対応など LLM が使えない場合は検索結果のみ返す
    if (!engine) {
        return {
            docs,
            answer:
              "この環境では WebGPU が利用できないため、回答生成（LLM）はスキップしました。上位5件の関連データを表示します。"
        };
    }

    const context =
      docs
      .map(d => `${d.title}\n${d.text}\nURL: ${d.url}`)
      .join("\n\n");

    const prompt = `
以下の情報のみ利用して回答してください。

${context}

質問:
${question}
`;

    const response =
      await engine.chat.completions.create({
        messages: [
          {
            role: "user",
            content: prompt
          }
        ]
      });

    return {
        docs,
        answer: response.choices[0]
            .message.content
    };
}

/*
----------------------------------------
UI
----------------------------------------
*/

document
.getElementById("askButton")
.addEventListener(
"click",
async () => {

    const question =
      document
      .getElementById("question")
      .value;

    const resultsDiv =
      document
      .getElementById("results");

    const answerDiv =
      document
      .getElementById("answer");

    resultsDiv.innerHTML = "";
    answerDiv.textContent =
      "検索中...";

    try {

        const { docs, answer } =
          await ask(question);

        renderResults(docs);

        answerDiv.textContent =
          answer
            ? answer
            : "（この環境ではWebGPUが利用できないため、AIによる回答生成はスキップし、検索結果のみ表示しています）";

    } catch(err) {

        console.error(err);

        answerDiv.textContent =
          err.message;
    }
});

function renderResults(docs) {

    const resultsDiv =
      document
      .getElementById("results");

    if (!docs.length) {
        resultsDiv.innerHTML =
          "<p>該当するデータが見つかりませんでした。</p>";
        return;
    }

    const items = docs.map((d, i) => {

        const score =
          (d.score * 100).toFixed(1);

        const title =
          d.title || "(タイトル不明)";

        const titleHtml =
          d.url
            ? `<a href="${d.url}" target="_blank" rel="noopener">${i + 1}. ${escapeHtml(title)}</a>`
            : `<span>${i + 1}. ${escapeHtml(title)}</span>`;

        return `
          <li>
            ${titleHtml}
            <div class="meta">
              ${escapeHtml(d.organization || "")}
              <span class="score">関連度 ${score}%</span>
            </div>
          </li>`;
    }).join("");

    resultsDiv.innerHTML =
      `<h2>関連データ 上位5件</h2><ol class="hits">${items}</ol>`;
}

function escapeHtml(str) {
    if (str === null || str === undefined) {
        return "";
    }
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

/*
----------------------------------------
Startup
----------------------------------------
*/

(async () => {

    document
    .getElementById("answer")
    .textContent =
    "Embedding初期化中...";

    await initializeEmbedding();

    document
    .getElementById("answer")
    .textContent =
    "インデックス作成中...";

    await buildIndex((done, total) => {
        document
        .getElementById("answer")
        .textContent =
        `インデックス作成中... (${done}/${total})`;
    });

    if (hasWebGPU()) {

        document
        .getElementById("answer")
        .textContent =
        "LLMロード中...(数GBダウンロードされます)";

        try {
            await initializeLLM();
        } catch(err) {
            console.warn("LLM初期化に失敗しました。検索のみで動作します。", err);
            engine = undefined;
        }
    } else {
        console.warn("WebGPU が利用できないため、AI回答生成を無効化し検索のみで動作します。");
    }

    document
    .getElementById("answer")
    .textContent =
    hasWebGPU() && engine
        ? "準備完了"
        : "準備完了（検索のみ・AI回答生成は無効）";
})();
