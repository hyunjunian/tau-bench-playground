/**
 * Full exports of finished episodes, kept in IndexedDB (localStorage is too
 * small for transcripts) so History entries can be reopened. Everything here
 * is best effort: without IndexedDB the history simply has nothing to reopen.
 */

const DB_NAME = "tau-playground";
const STORE = "episodes";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  dbPromise ??= new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof indexedDB === "undefined") return reject(new Error("IndexedDB is not available"));
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
  }).catch((e: unknown) => {
    dbPromise = null;
    throw e;
  });
  return dbPromise;
}

function request<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const req = run(db.transaction(STORE, mode).objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
      }),
  );
}

/** Store an episode export (JSON text) under a history entry id. */
export function archiveEpisode(id: string, json: string): Promise<void> {
  return request("readwrite", (s) => s.put(json, id)).then(() => undefined);
}

export function loadArchivedEpisode(id: string): Promise<string | null> {
  return request<unknown>("readonly", (s) => s.get(id)).then((v) => (typeof v === "string" ? v : null));
}

/** Delete every stored episode whose id is not in `keep`. */
export async function pruneArchive(keep: Set<string>): Promise<void> {
  const keys = await request<IDBValidKey[]>("readonly", (s) => s.getAllKeys());
  await Promise.all(keys.filter((k) => !keep.has(String(k))).map((k) => request("readwrite", (s) => s.delete(k))));
}

export function clearArchive(): Promise<void> {
  return request("readwrite", (s) => s.clear()).then(() => undefined);
}
