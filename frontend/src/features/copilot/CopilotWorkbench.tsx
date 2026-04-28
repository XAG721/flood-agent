import { FormEvent, KeyboardEvent, useState, type ComponentProps } from "react";
import styles from "../../App.module.css";
import { CommandCenterPage } from "../../components/CommandCenterPage";
import { quickPrompts } from "../../config/consoleConfig";

type CommandCenterPageProps = ComponentProps<typeof CommandCenterPage>;

type CopilotWorkbenchProps = Omit<
  CommandCenterPageProps,
  "input" | "quickPrompts" | "onChangeInput" | "onPrompt" | "onSubmit" | "onTextareaKeyDown"
> & {
  onAsk: (prompt: string) => void;
};

export function CopilotWorkbench({ onAsk, ...commandCenterProps }: CopilotWorkbenchProps) {
  const [input, setInput] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!input.trim()) return;
    onAsk(input.trim());
    setInput("");
  }

  function handleTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <>
      <div className={styles.panelFrame}>
        <div className={styles.panelHeaderCompact}>
          <div>
            <p className={styles.sectionLabel}>对话指挥</p>
            <h2>通过对话查看研判、请示与总结</h2>
          </div>
        </div>
        <p className={styles.emptyState}>
          在这里可以查看智能体的多轮分析、审批请求、日报和高风险复盘，并继续追问对象风险、联动建议和执行策略。
        </p>
        <div className={styles.routeSummary}>
          <div>
            <span>使用方式</span>
            <strong>输入问题或使用快捷提示，系统会结合事件上下文持续生成建议。</strong>
          </div>
        </div>
      </div>
      <CommandCenterPage
        {...commandCenterProps}
        input={input}
        quickPrompts={quickPrompts}
        onChangeInput={setInput}
        onPrompt={onAsk}
        onSubmit={handleSubmit}
        onTextareaKeyDown={handleTextareaKeyDown}
      />
    </>
  );
}
