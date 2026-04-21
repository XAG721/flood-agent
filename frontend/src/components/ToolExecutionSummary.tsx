import storyStyles from "../styles/story-panels.module.css";
import type { StructuredAnswer } from "../types/api";

function toolStatusLabel(status: NonNullable<StructuredAnswer["tool_executions"]>[number]["status"]) {
  return {
    success: "成功",
    failed: "失败",
    skipped: "跳过",
    timeout: "超时",
  }[status];
}

function toolStatusClass(status: NonNullable<StructuredAnswer["tool_executions"]>[number]["status"]) {
  return {
    success: storyStyles.toolStatusSuccess,
    failed: storyStyles.toolStatusFailed,
    skipped: storyStyles.toolStatusSkipped,
    timeout: storyStyles.toolStatusTimeout,
  }[status];
}

interface ToolExecutionSummaryProps {
  answer: StructuredAnswer | null;
}

export function ToolExecutionSummary({ answer }: ToolExecutionSummaryProps) {
  const executions = answer?.tool_executions ?? [];

  return (
    <section className={storyStyles.toolSummary}>
      <div className={storyStyles.toolHeader}>
        <div>
          <p className={storyStyles.eyebrow}>关键工具调用</p>
          <h3>执行纪要</h3>
          <p>只保留真正推动本轮处置的能力调用，用结果和状态说明价值，不把底层日志全部抬到首屏。</p>
        </div>
      </div>

      <div className={storyStyles.toolList}>
        {executions.length ? (
          executions.slice(0, 4).map((item) => (
            <article
              key={`${item.execution_id ?? item.tool_name}_${item.status}_${item.duration_ms}`}
              className={storyStyles.toolCard}
            >
              <div className={storyStyles.toolTopline}>
                <div>
                  <strong>{item.tool_name}</strong>
                  <p>{item.output_summary}</p>
                </div>
                <span className={`${storyStyles.toolStatus} ${toolStatusClass(item.status)}`}>
                  {toolStatusLabel(item.status)}
                </span>
              </div>
              <div className={storyStyles.toolMeta}>
                <span>{item.duration_ms} ms</span>
                {item.parallel_group ? <span>并行组 {item.parallel_group}</span> : null}
                {item.cache_hit ? <span>命中缓存</span> : null}
                {item.fallback_from_tool ? <span>由 {item.fallback_from_tool} 兜底</span> : null}
              </div>
            </article>
          ))
        ) : (
          <p>当前还没有可展示的工具调用纪要，生成处置建议后这里会自动补齐。</p>
        )}
      </div>
    </section>
  );
}
