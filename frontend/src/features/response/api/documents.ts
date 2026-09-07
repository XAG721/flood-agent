import { request } from "../../../lib/httpClient";
import type { DocumentVersionRecord, IndexBuildRecord, WorkflowRole } from "../../../types/response";
import { actionPayload } from "./shared";

export const responseDocumentApi = {
  listDocuments(): Promise<DocumentVersionRecord[]> {
    return request("/response/documents", { method: "GET" });
  },
  registerDocument(input: {
    document_id: string;
    title: string;
    version_label: string;
    issuer: string;
    jurisdiction: string;
    effective_at: string;
    content: string;
    replaces_version_id?: string;
  }): Promise<DocumentVersionRecord> {
    return request("/response/documents", {
      method: "POST",
      body: JSON.stringify({
        ...input,
        ...actionPayload("admin", "登记并索引受控模拟预案版本"),
      }),
    });
  },
  createDocumentVersion(input: {
    documentId: string;
    title: string;
    versionLabel: string;
    issuer: string;
    jurisdiction: string;
    effectiveAt: string;
    expiresAt?: string;
    content?: string;
    file?: { filename: string; mediaType: string; contentBase64: string; sha256: string };
    ocrText?: string;
    ocrEngineVersion?: string;
    replacesVersionId?: string;
    operatorRole: WorkflowRole;
  }): Promise<DocumentVersionRecord> {
    return request(`/response/documents/${encodeURIComponent(input.documentId)}/versions`, {
      method: "POST",
      body: JSON.stringify({
        title: input.title,
        version_label: input.versionLabel,
        issuer: input.issuer,
        jurisdiction: input.jurisdiction,
        effective_at: input.effectiveAt,
        expires_at: input.expiresAt,
        content: input.content,
        file: input.file ? {
          filename: input.file.filename,
          media_type: input.file.mediaType,
          content_base64: input.file.contentBase64,
          sha256: input.file.sha256,
        } : undefined,
        ocr_text: input.ocrText,
        ocr_engine_version: input.ocrEngineVersion,
        replaces_version_id: input.replacesVersionId,
        auto_publish: false,
        is_simulated: true,
        ...actionPayload(input.operatorRole, "登记不可变文档源文件版本"),
      }),
    });
  },
  parseDocumentVersion(versionId: string, operatorRole: WorkflowRole, expectedVersion: number): Promise<DocumentVersionRecord> {
    return request(`/response/document-versions/${encodeURIComponent(versionId)}/parse`, {
      method: "POST",
      body: JSON.stringify({
        parser_version: "document-parser-v2",
        expected_version: expectedVersion,
        ...actionPayload(operatorRole, "解析页码、章节、条款和表格定位"),
      }),
    });
  },
  publishDocumentVersion(versionId: string, operatorRole: WorkflowRole, expectedVersion: number): Promise<DocumentVersionRecord> {
    return request(`/response/document-versions/${encodeURIComponent(versionId)}/publish`, {
      method: "POST",
      body: JSON.stringify({
        expected_version: expectedVersion,
        corpus_version: "district-policy-corpus-v1",
        embedding_version: "deterministic-hybrid-v1",
        ...actionPayload(operatorRole, "复核通过并发布当前文档版本"),
      }),
    });
  },
  retireDocumentVersion(versionId: string, operatorRole: WorkflowRole, expectedVersion: number, reason: string): Promise<DocumentVersionRecord> {
    return request(`/response/document-versions/${encodeURIComponent(versionId)}/retire`, {
      method: "POST",
      body: JSON.stringify({
        reason,
        expected_version: expectedVersion,
        ...actionPayload(operatorRole, reason),
      }),
    });
  },
  rebuildDocumentIndex(versionId: string, operatorRole: WorkflowRole): Promise<IndexBuildRecord> {
    return request("/response/index-builds", {
      method: "POST",
      body: JSON.stringify({
        document_version_id: versionId,
        ...actionPayload(operatorRole, "重建发布文档的可复现索引"),
      }),
    });
  },
};
