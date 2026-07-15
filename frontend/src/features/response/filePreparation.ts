function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

export async function prepareRegistryFile(file: File) {
  if (file.size > 10 * 1024 * 1024) {
    throw new Error("风险对象文件不得超过 10 MiB");
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error("当前浏览器不支持文件 SHA-256 校验");
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes));
  const sha256 = Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
  const suffix = file.name.split(".").pop()?.toLowerCase();
  const mediaType = file.type || {
    csv: "text/csv",
    xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    json: "application/json",
    geojson: "application/geo+json",
  }[suffix ?? ""];
  if (!mediaType) {
    throw new Error("只支持 CSV、XLSX、JSON 或 GeoJSON 风险对象文件");
  }
  return {
    filename: file.name,
    mediaType,
    contentBase64: bytesToBase64(bytes),
    sha256,
  };
}

export async function prepareDocumentFile(file: File) {
  if (file.size > 10 * 1024 * 1024) {
    throw new Error("文档文件不得超过 10 MiB");
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error("当前浏览器不支持文件 SHA-256 校验");
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes));
  const sha256 = Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
  const suffix = file.name.split(".").pop()?.toLowerCase();
  const mediaType = file.type || {
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    csv: "text/csv",
    txt: "text/plain",
    md: "text/markdown",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    tif: "image/tiff",
    tiff: "image/tiff",
  }[suffix ?? ""];
  if (!mediaType) {
    throw new Error("只支持 PDF、DOCX、XLSX、CSV、TXT、Markdown 或扫描图片");
  }
  return {
    filename: file.name,
    mediaType,
    contentBase64: bytesToBase64(bytes),
    sha256,
    scanned: mediaType.startsWith("image/"),
  };
}
