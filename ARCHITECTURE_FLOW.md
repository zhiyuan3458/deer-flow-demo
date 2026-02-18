# DeerFlow Agent 架构流程图

## 一、整体架构流程图

```mermaid
graph TB
    Start([用户在 Web UI 输入问题]) --> FE[前端 Next.js]
    FE -->|HTTP POST /api/chat/stream| BE[后端 FastAPI Server]
    
    BE --> Graph[LangGraph 状态机]
    
    Graph --> Coordinator[Coordinator Node<br/>意图识别与澄清]
    
    Coordinator -->|需要澄清| Clarify{澄清模式}
    Clarify -->|继续提问| InterruptUI[中断并等待用户回答]
    InterruptUI --> Coordinator
    Clarify -->|澄清完成| BG
    
    Coordinator -->|直接回答| DirectEnd[直接响应用户]
    Coordinator -->|研究任务| BG{背景调查?}
    
    BG -->|启用| BGInv[Background Investigation<br/>初步搜索]
    BG -->|禁用| Planner
    BGInv --> Planner[Planner Node<br/>生成研究计划]
    
    Planner --> HumanFB{人工审核?}
    HumanFB -->|需要审核| Interrupt[中断并展示计划]
    Interrupt -->|ACCEPTED| ResearchTeam
    Interrupt -->|EDIT_PLAN| Planner
    HumanFB -->|自动接受| ResearchTeam
    
    Planner -->|足够上下文| Reporter
    
    ResearchTeam[Research Team Router] --> StepType{步骤类型}
    
    StepType -->|research| Researcher[Researcher Agent<br/>Web搜索+爬虫]
    StepType -->|processing| Coder[Coder Agent<br/>代码执行]
    StepType -->|analysis| Analyst[Analyst Agent<br/>纯推理分析]
    
    Researcher --> Tools1[工具调用]
    Tools1 --> WebSearch[Web Search<br/>Tavily/Brave/DuckDuckGo]
    Tools1 --> Crawler[Web Crawler<br/>Jina/InfoQuest]
    Tools1 --> RAG[RAG检索<br/>私有知识库]
    Tools1 --> MCP1[MCP服务<br/>动态工具]
    
    Coder --> Tools2[Python REPL<br/>代码执行]
    
    Analyst --> LLMReason[纯LLM推理<br/>无工具]
    
    Tools1 --> NextStep{还有步骤?}
    Tools2 --> NextStep
    LLMReason --> NextStep
    
    NextStep -->|是| ResearchTeam
    NextStep -->|否| Reporter[Reporter Node<br/>生成最终报告]
    
    Reporter --> SSE[SSE 流式返回]
    SSE --> Stream1[event: message_chunk]
    SSE --> Stream2[event: tool_calls]
    SSE --> Stream3[event: tool_call_result]
    SSE --> Stream4[event: interrupt]
    
    Stream1 --> FEProcess[前端事件处理器]
    Stream2 --> FEProcess
    Stream3 --> FEProcess
    Stream4 --> FEProcess
    
    FEProcess --> UIRender[动态 UI 渲染]
    UIRender --> UI1[Markdown 渲染报告]
    UIRender --> UI2[工具调用动画]
    UIRender --> UI3[实时消息流]
    UIRender --> UI4[交互式编辑器]
    
    DirectEnd --> FEProcess
    
    style Start fill:#e1f5ff
    style FE fill:#fff4e6
    style BE fill:#f3e5f5
    style Graph fill:#e8f5e9
    style Coordinator fill:#fff9c4
    style Planner fill:#fff9c4
    style Reporter fill:#fff9c4
    style Researcher fill:#ffccbc
    style Coder fill:#ffccbc
    style Analyst fill:#ffccbc
    style SSE fill:#e1bee7
    style UIRender fill:#c5e1a5
```

## 二、数据流向时序图

```mermaid
sequenceDiagram
    participant U as 用户
    participant FE as 前端 Next.js
    participant API as FastAPI Server
    participant G as LangGraph
    participant C as Coordinator
    participant P as Planner
    participant R as Researcher
    participant Rep as Reporter
    participant T as 外部工具

    U->>FE: 输入问题
    FE->>API: POST /api/chat/stream
    API->>G: 初始化工作流
    G->>C: 启动 Coordinator
    
    alt 需要澄清
        C-->>API: SSE interrupt事件
        API-->>FE: 显示澄清问题
        FE-->>U: 等待用户回答
        U->>FE: 提供更多信息
        FE->>API: 继续对话
        API->>C: 处理回答
    end
    
    C->>P: handoff_to_planner
    P->>P: 生成研究计划
    
    alt 需要人工审核
        P-->>API: SSE interrupt事件
        API-->>FE: 展示计划
        FE-->>U: 请求批准
        U->>FE: ACCEPTED 或 EDIT_PLAN
    end
    
    P->>R: 执行研究步骤
    
    loop 每个研究步骤
        R->>T: 调用工具 搜索/爬虫/RAG
        R-->>API: SSE tool_calls
        API-->>FE: 显示工具调用动画
        
        T->>R: 返回结果
        R-->>API: SSE tool_call_result
        API-->>FE: 显示结果
    end
    
    R->>Rep: 所有步骤完成
    Rep->>Rep: 汇总生成报告
    
    loop 流式生成报告
        Rep-->>API: SSE message_chunk
        API-->>FE: 实时显示报告内容
        FE-->>U: 渐进式渲染Markdown
    end
    
    Rep-->>API: SSE 完成标志
    API-->>FE: 关闭SSE连接
```

## 三、Agent节点详细流程图

```mermaid
graph LR
    subgraph Coordinator流程
        C1[接收用户输入] --> C2{判断意图}
        C2 -->|研究任务| C3[调用handoff_to_planner]
        C2 -->|闲聊问候| C4[调用direct_response]
        C2 -->|需要澄清| C5[多轮对话收集信息]
        C5 --> C6{达到最大轮次?}
        C6 -->|否| C5
        C6 -->|是| C3
    end
    
    subgraph Planner流程
        P1[接收研究主题] --> P2[调用LLM生成计划]
        P2 --> P3[解析JSON计划]
        P3 --> P4{验证计划}
        P4 -->|无效| P5[修复计划]
        P5 --> P3
        P4 -->|有效| P6{有足够上下文?}
        P6 -->|是| P7[直接到Reporter]
        P6 -->|否| P8[到Human Feedback]
    end
    
    subgraph Researcher流程
        R1[接收研究步骤] --> R2[准备工具列表]
        R2 --> R3{Web搜索启用?}
        R3 -->|是| R4[添加搜索工具]
        R3 -->|否| R5[跳过搜索工具]
        R4 --> R6[添加RAG工具]
        R5 --> R6
        R6 --> R7[添加MCP工具]
        R7 --> R8[创建Agent]
        R8 --> R9[执行Agent]
        R9 --> R10[收集结果]
    end
    
    subgraph Reporter流程
        Rep1[接收所有观察结果] --> Rep2[上下文压缩]
        Rep2 --> Rep3[调用LLM生成报告]
        Rep3 --> Rep4[流式返回报告]
    end
```

## 四、工具调用流程图

```mermaid
graph TD
    A[Agent决定调用工具] --> B{工具类型}
    
    B -->|Web Search| C1[Tavily API]
    B -->|Web Search| C2[Brave Search]
    B -->|Web Search| C3[DuckDuckGo]
    
    B -->|Crawler| D1[Jina Reader]
    B -->|Crawler| D2[InfoQuest]
    
    B -->|RAG| E1[Qdrant]
    B -->|RAG| E2[Milvus]
    B -->|RAG| E3[RAGFlow]
    B -->|RAG| E4[VikingDB]
    
    B -->|Code| F1[Python REPL]
    
    B -->|MCP| G1[动态加载MCP工具]
    
    C1 --> H[返回结果]
    C2 --> H
    C3 --> H
    D1 --> H
    D2 --> H
    E1 --> H
    E2 --> H
    E3 --> H
    E4 --> H
    F1 --> H
    G1 --> H
    
    H --> I[更新State]
    I --> J[继续下一步]
```

## 五、前端SSE事件处理流程

```mermaid
graph TD
    A[前端发起SSE连接] --> B[监听事件流]
    
    B --> C{事件类型}
    
    C -->|message_chunk| D1[追加文本到报告]
    C -->|tool_calls| D2[显示工具调用动画]
    C -->|tool_call_result| D3[显示工具执行结果]
    C -->|interrupt| D4[显示交互界面]
    C -->|error| D5[显示错误信息]
    
    D1 --> E[更新UI]
    D2 --> E
    D3 --> E
    D4 --> F{用户操作}
    F -->|ACCEPTED| G[发送反馈到后端]
    F -->|EDIT_PLAN| G
    G --> B
    
    D5 --> E
    
    E --> H{流结束?}
    H -->|否| B
    H -->|是| I[关闭连接]
```

## 六、状态转换图

```mermaid
stateDiagram-v2
    [*] --> Coordinator: 用户输入
    
    Coordinator --> BackgroundInvestigator: 启用背景调查
    Coordinator --> Planner: 直接到规划
    Coordinator --> End: 直接回答
    
    BackgroundInvestigator --> Planner: 完成背景调查
    
    Planner --> HumanFeedback: 需要审核
    Planner --> Reporter: 有足够上下文
    
    HumanFeedback --> Planner: EDIT_PLAN
    HumanFeedback --> ResearchTeam: ACCEPTED
    
    ResearchTeam --> Researcher: research步骤
    ResearchTeam --> Coder: processing步骤
    ResearchTeam --> Analyst: analysis步骤
    ResearchTeam --> Planner: 需要重新规划
    ResearchTeam --> Reporter: 所有步骤完成
    
    Researcher --> ResearchTeam: 完成研究
    Coder --> ResearchTeam: 完成编码
    Analyst --> ResearchTeam: 完成分析
    
    Reporter --> [*]: 生成最终报告
```

## 七、核心文件路径结构

```
后端核心架构
├── src/server/app.py                    # FastAPI 主服务器
│   ├── /api/chat/stream                 # 核心SSE流接口
│   ├── /api/tts                         # 文本转语音
│   ├── /api/podcast/generate            # 播客生成
│   └── /api/ppt/generate                # PPT生成
│
├── src/graph/                           # LangGraph工作流
│   ├── builder.py                       # 状态图构建器
│   ├── nodes.py                         # 核心Agent节点实现
│   ├── types.py                         # 状态类型定义
│   └── utils.py                         # 工具函数
│
├── src/agents/agents.py                 # Agent创建工厂
├── src/tools/                           # 工具集
│   ├── search.py                        # 搜索工具
│   ├── crawler.py                       # 爬虫工具
│   └── python_repl.py                   # 代码执行
│
├── src/rag/                             # RAG集成
│   ├── retriever.py                     # 检索器
│   └── builder.py                       # RAG构建器
│
└── src/prompts/                         # Prompt模板
    └── *.md                             # 各Agent的提示词

前端核心架构
├── web/src/app/chat/
│   ├── page.tsx                         # 聊天页面入口
│   ├── main.tsx                         # 主组件
│   └── components/
│       ├── message-list-view.tsx        # 消息列表
│       ├── research-block.tsx           # 研究过程展示
│       └── input-box.tsx                # 输入框组件
│
├── web/src/core/
│   ├── api/chat.ts                      # 聊天API封装
│   ├── sse/
│   │   ├── StreamEvent.ts               # SSE事件类型
│   │   └── fetch-stream.ts              # SSE流处理
│   └── messages/
│       ├── types.ts                     # 消息类型定义
│       └── merge-message.ts             # 消息合并逻辑
│
└── web/src/components/
    ├── editor/                          # Notion风格编辑器
    └── deer-flow/markdown.tsx           # Markdown渲染器
```

