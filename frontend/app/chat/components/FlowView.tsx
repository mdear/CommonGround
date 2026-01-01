import React, { useMemo, useRef, useLayoutEffect, useCallback } from 'react';
import { Badge } from '@/components/ui/badge';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  Position,
  Handle,
  NodeMouseHandler,
  NodeProps,
  EdgeProps,
  BaseEdge,
  useReactFlow,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useFlowView } from '../hooks/useFlowView';
import { sessionStore, FlowNodeData } from '@/app/stores/sessionStore';
import { observer } from 'mobx-react-lite';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import LoadingSpinner from '@/components/layout/LoadingSpinner';

// Custom edge component - bend point is unified at 25px before the endpoint
const CustomEdge = ({ id, sourceX, sourceY, targetX, targetY, style }: EdgeProps) => {
  // Calculate the bend point position: 25px before the target
  const bendOffset = 15;
  let bendY = targetY - bendOffset;

  // Ensure the bend point is not too close to the source
  if (Math.abs(bendY - sourceY) < 20) {
    bendY = sourceY + (targetY - sourceY) * 0.7; // If it's too close, use 70% of the position
  }

  // Create path: source -> bend point -> target
  const path = `M ${sourceX},${sourceY} L ${sourceX},${bendY} L ${targetX},${bendY} L ${targetX},${targetY}`;

  return (
    <BaseEdge
      id={id}
      path={path}
      style={{
        ...style,
        stroke: '#888888',
        strokeWidth: 1,
        fill: 'none'
      }}
    />
  );
};

const formatTime = (timestamp?: string) => {
  if (!timestamp) return '';
  return new Date(timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

const ParamsList = ({ data }: { data: Record<string, unknown> | null | undefined }) => {
  const renderValue = (value: unknown): React.ReactNode => {
    if (typeof value !== 'object' || value === null) {
      return <span className="font-mono bg-white px-1 rounded">{String(value)}</span>;
    }
    return <ParamsList data={value as Record<string, unknown>} />;
  };

  if (Array.isArray(data)) {
    return (
      <ul className="list-disc pl-4 mt-0.5">
        {data.map((item, index) => <li key={index}>{renderValue(item)}</li>)}
      </ul>
    );
  }

  // It's an object
  const entries = data ? Object.entries(data) : [];

  if (entries.length === 0) {
    return <div className="text-gray-400">No parameters</div>;
  }

  return (
    <div className="space-y-1">
      {entries.map(([key, value]) => (
        <div key={key} className="flex items-start">
          <span className="font-semibold text-gray-600 w-24 flex-shrink-0">{key}:</span>
          <div className="ml-2 flex-grow">{renderValue(value)}</div>
        </div>
      ))}
    </div>
  );
};

// Calculate content box height level based on content length
const getContentHeightClass = (content: string): { class: string; size: string } => {
  if (!content) return { class: 'h-0', size: 'XS' }; // XS - height is 0 for no content

  const length = content.length;
  if (length <= 50) return { class: 'h-6', size: 'S' }; // S - 1 line of text
  if (length <= 200) return { class: 'h-20', size: 'M' }; // M - 5 lines of text
  if (length <= 400) return { class: 'h-36', size: 'L' }; // L - 9 lines of text
  if (length <= 800) return { class: 'h-52', size: 'XL' }; // XL - 13 lines of text
  return { class: 'h-80', size: 'XXL' }; // XXL - 20 lines of text (capped for very long content)
};

const TurnNodeContent = observer(({ data }: { data: FlowNodeData }) => {
  const streamingContent = data.content_stream_id ? sessionStore.streamingContent.get(data.content_stream_id) : '';

  // When running, prioritize streaming content. When finished, show only final content.
  const displayContent = data.status === 'running'
    ? (streamingContent || data.final_content || '')
    : (data.final_content || '');

  const isRunningButEmpty = data.status === 'running' && !displayContent;
  const hasTools = data.tool_interactions && data.tool_interactions.length > 0;
  const hasContent = isRunningButEmpty || displayContent;

  // Dynamically calculate the height level of the current content (including streaming content)
  const currentContentHeight = getContentHeightClass(displayContent);

  // Get the layer's preset max content level as the minimum height
  const layerMaxLevel = data.layerMaxContentLevel || 'XS';

  // Get height class based on the max level for the layer
  const getHeightClassForLevel = (level: string) => {
    switch (level) {
      case 'XS': return 'h-0';
      case 'S': return 'h-6';
      case 'M': return 'h-20';
      case 'L': return 'h-36';
      case 'XL': return 'h-52';
      case 'XXL': return 'h-80';
      default: return 'h-0';
    }
  };

  // Take the maximum of the current content's required height and the layer's minimum height
  const levels = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];
  const currentLevelIndex = levels.indexOf(currentContentHeight.size);
  const layerLevelIndex = levels.indexOf(layerMaxLevel);
  const finalLevel = levels[Math.max(currentLevelIndex, layerLevelIndex)];

  const unifiedContentHeight = {
    class: getHeightClassForLevel(finalLevel),
    size: finalLevel
  };

  return (
    <div className="flex-1 flex flex-col text-left min-h-0">
      {/* Header */}
      <div className="p-2 flex-shrink-0">
        <div className="inline-flex items-center gap-2 rounded-full p-1 pr-2 text-black">
          <div className="w-5 h-5 bg-gray-300 rounded-full flex items-center justify-center text-xs font-bold">
            {data.label.charAt(0).toUpperCase()}
          </div>
          <span className="font-medium text-sm">{data.label}</span>
        </div>
      </div>

      {/* Main content area - use unified height level for the layer */}
      <div className={`mx-2 mb-2 text-sm text-gray-700 flex-shrink-0 bg-white rounded-lg p-4 overflow-y-auto relative ${unifiedContentHeight.class} ${hasContent ? 'opacity-100' : 'opacity-0'}`}>
        {isRunningButEmpty ? (
          <div className="flex items-center">
            <span className="text-gray-400 italic">Thinking...</span>
          </div>
        ) : displayContent ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>
        ) : (
          <div className="flex items-center">
            <span className="text-gray-400 italic">No content</span>
          </div>
        )}
      </div>

      {/* Tool Interactions Section - always visible, transparent when no tools */}
      <div className={`flex-shrink-0 border-gray-200 mx-2 ${hasTools ? 'opacity-100' : 'opacity-0'}`}>
        <div className="max-h-[240px] overflow-y-auto p-1 space-y-1">
          {hasTools ? data.tool_interactions?.map(interaction => (
            <div key={interaction.tool_call_id} className="text-sm p-1">
              <div className="flex items-center justify-between w-full">
                <span className="font-medium truncate" title={interaction.tool_name}>
                  {interaction.tool_name || 'Tool Call'}
                </span>
                <Badge
                  variant={
                    interaction.status === 'running'
                      ? 'default'
                      : interaction.status === 'completed_success'
                      ? 'secondary'
                      : interaction.status === 'completed_error'
                      ? 'destructive'
                      : 'outline'
                  }
                  className="text-xs ml-2"
                >
                  {interaction.status}
                </Badge>
              </div>
              {!!interaction.input_params?.purpose && (
                <p className="text-xs text-gray-600 bg-gray-100 p-1 rounded mt-1 whitespace-pre-wrap break-all">
                  {String(interaction.input_params.purpose)}
                </p>
              )}
            </div>
          )) : (
            <div className="text-sm p-1">
              <div className="flex items-center justify-between w-full">
                <span className="font-medium truncate">No tools</span>
                <Badge variant="outline" className="text-xs ml-2">
                  idle
                </Badge>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Timestamp footer */}
      {data.timestamp && (
        <div className="text-xs text-gray-400 mt-auto px-3 pb-1 text-right flex-shrink-0">
          {formatTime(data.timestamp)}
        </div>
      )}
    </div>
  );
});

const DefaultNodeContent = ({ data }: { data: FlowNodeData }) => (
  <div className="text-center">
    <div className="text-sm font-bold truncate">{data.label}</div>
    {data.timestamp && (
      <div className="text-xs opacity-70 mt-1">{formatTime(data.timestamp)}</div>
    )}
  </div>
);

interface CustomNodeProps extends NodeProps<FlowNodeData> {
  onSizeChange: (id: string, size: { width: number; height: number }) => void;
}

const CustomNode = observer(({ id, data, onSizeChange }: CustomNodeProps) => {
  const nodeRef = useRef<HTMLDivElement>(null);
  const nodeType = data.nodeType || 'default';

  useLayoutEffect(() => {
    const nodeElement = nodeRef.current;
    if (!nodeElement) return;

    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        onSizeChange(id, { width, height });
      }
    });

    observer.observe(nodeElement);
    return () => observer.disconnect();
  }, [id, onSizeChange]);

  const nodeStyles: Record<string, string> = {
    turn: 'bg-gray-50',
    default: 'bg-white',
  };

  const statusStyles: Record<string, string> = {
    running: 'border-dashed border-blue-400 animate-pulse',
    completed_error: 'border-solid border-red-400',
    interrupted: 'border-dashed border-gray-400',
    default: 'border-solid border-gray-200',
  };

  if (nodeType === 'gather') {
    return (
      <div ref={nodeRef} style={{ width: 340, height: 35 }} className="flex items-center justify-center relative">
        <Handle type="target" position={Position.Top} className="!w-2 !h-2" />

        {/* Gather node - borderless "Synthesis" text, aligned with other cards' width */}
        <div className="w-full h-full flex items-center justify-center">
          <span className="text-sm font-medium text-gray-700 select-none">
            Synthesis
          </span>
        </div>
        <Handle type="source" position={Position.Bottom} className="!w-2 !h-2" />
      </div>
    );
  }

  if (nodeType === 'epoch_separator') {
    return (
      <div ref={nodeRef} style={{ width: 340, height: 40 }} className="flex items-center justify-center relative">
        <Handle type="target" position={Position.Top} className="!w-2 !h-2" />

        {/* Epoch separator - horizontal line with epoch label */}
        <div className="w-full h-full flex items-center justify-center gap-3">
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-blue-300 to-blue-400"></div>
          <span className="text-xs font-medium text-blue-500 select-none whitespace-nowrap px-2 py-1 bg-blue-50 rounded-full border border-blue-200">
            {data.label || 'New Epoch'}
          </span>
          <div className="flex-1 h-px bg-gradient-to-l from-transparent via-blue-300 to-blue-400"></div>
        </div>
        <Handle type="source" position={Position.Bottom} className="!w-2 !h-2" />
      </div>
    );
  }

  const containerClasses = `
    shadow-md rounded-lg border-2 overflow-hidden flex flex-col relative
    ${nodeStyles[nodeType] || nodeStyles.default}
    ${statusStyles[data.status] || statusStyles.default}
  `;

  const renderContent = () => {
    if (nodeType === 'turn') {
      return <TurnNodeContent data={data} />;
    }
    return <DefaultNodeContent data={data} />;
  };

  return (
    <div ref={nodeRef} style={{ width: 340 }} className={containerClasses}>
      <Handle type="target" position={Position.Top} className="!w-2 !h-2" style={{ left: '50%', transform: 'translateX(-50%)' }} />

      {renderContent()}
      <Handle type="source" position={Position.Bottom} className="!w-2 !h-2" style={{ left: '50%', transform: 'translateX(-50%)' }} />
    </div>
  );
});
CustomNode.displayName = 'CustomNode';

interface FlowViewProps {
  onNodeClick: NodeMouseHandler;
}

// Inner component that can use useReactFlow hook
const FlowViewInner = observer(({ onNodeClick }: FlowViewProps) => {
  const { nodes, edges, onNodeSizesChange } = useFlowView(sessionStore.flowStructure);
  const proOptions = { hideAttribution: true };

  // Used to track if fitView should be called (only on first data receipt)
  const [shouldFitView, setShouldFitView] = React.useState(false);

  // Listen for ViewModel state changes
  React.useEffect(() => {
    if (sessionStore.flowStructure && sessionStore.isWaitingForNewViewModel === false) {
      // Set fitView on first data arrival
      setShouldFitView(true);
      // Set a short delay to reset fitView state, ensuring it triggers only once
      const timer = setTimeout(() => setShouldFitView(false), 100);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionStore.flowStructure, sessionStore.isWaitingForNewViewModel]);

  const nodeTypes = useMemo(
    () => ({ custom: (props: NodeProps<FlowNodeData>) => <CustomNode {...props} onSizeChange={onNodeSizesChange} /> }),
    [onNodeSizesChange]
  );

  const edgeTypes = useMemo(
    () => ({
      custom: CustomEdge
    }),
    []
  );

  // Calculate the bounding box of all nodes
  const nodeBounds = useMemo(() => {
    if (nodes.length === 0) {
      return { minX: 0, minY: 0, maxX: 1000, maxY: 800, width: 1000, height: 800 };
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    nodes.forEach(node => {
      const x = node.position.x;
      const y = node.position.y;
      const width = node.width || 380;
      const height = node.height || 200;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + width);
      maxY = Math.max(maxY, y + height);
    });

    return {
      minX, minY, maxX, maxY,
      width: maxX - minX,
      height: maxY - minY
    };
  }, [nodes]);

  // Calculate translate extent to constrain panning within reasonable bounds around nodes
  // This prevents users from panning into infinite empty space, which causes minimap zoom-out
  const translateExtent = useMemo((): [[number, number], [number, number]] => {
    const padding = 500;
    return [
      [nodeBounds.minX - padding, nodeBounds.minY - padding],
      [nodeBounds.maxX + padding, nodeBounds.maxY + padding]
    ];
  }, [nodeBounds]);

  // Calculate dynamic minZoom based on content size
  // This ensures "zoom out all the way" shows all cards
  const dynamicMinZoom = useMemo(() => {
    // Assume a typical viewport of ~800x600 for calculation
    // The actual fitView will handle the real viewport
    const viewportWidth = 1200;
    const viewportHeight = 800;
    const padding = 0.2; // 20% padding around content

    const scaleX = viewportWidth / (nodeBounds.width * (1 + padding));
    const scaleY = viewportHeight / (nodeBounds.height * (1 + padding));
    const fitZoom = Math.min(scaleX, scaleY);

    // Don't go below 0.05 (5%) or above 0.5 for minZoom
    return Math.max(0.05, Math.min(0.5, fitZoom * 0.8));
  }, [nodeBounds]);

  const viewError = sessionStore.viewErrors.get('flow_view');

  if (viewError) {
    return (
      <div className="flex items-center justify-center h-full text-red-500 p-4 text-center">
        Error loading flow view: {viewError}
      </div>
    );
  }

  // If waiting for a new ViewModel or if flowStructure is null, show loading state
  if (sessionStore.isWaitingForNewViewModel || sessionStore.flowStructure === null) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner />
      </div>
    );
  }

  // maxZoom of 1.5 allows reading card text comfortably (150% of actual size)
  const maxZoom = 1.5;

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      defaultEdgeOptions={{ type: 'custom' }}
      fitView={shouldFitView}
      fitViewOptions={{
        padding: 0.3,  // 30% padding around content
        maxZoom: 1.0,  // fitView won't zoom past 100%
        minZoom: dynamicMinZoom
      }}
      defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
      minZoom={dynamicMinZoom}  // Dynamic: adapts to show all cards
      maxZoom={maxZoom}         // Fixed: readable card text at 150%
      translateExtent={translateExtent}
      className="bg-white rounded-lg border border-[#E4E4E4]"
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={true}
      panOnDrag={true}
      zoomOnScroll={true}       // Default canvas zoom (~9 clicks min to max)
      zoomOnDoubleClick={false}
      preventScrolling={true}
      proOptions={proOptions}
      onNodeClick={onNodeClick}
      nodesFocusable={true}
      edgesFocusable={false}
      selectNodesOnDrag={false}
    >
      <Background color="#ffffff" variant={BackgroundVariant.Dots} gap={12} size={1} />
      <Controls showZoom={true} showFitView={true} showInteractive={false} />
      <MiniMap
        nodeColor={(n) => {
          // Color nodes based on their status for better visibility
          const status = n.data?.status;
          if (status === 'running') return '#3b82f6'; // blue
          if (status === 'completed_success') return '#22c55e'; // green
          if (status === 'completed_error') return '#ef4444'; // red
          return '#94a3b8'; // slate for idle/default
        }}
        nodeStrokeColor="#64748b"
        nodeStrokeWidth={2}
        nodeBorderRadius={4}
        maskColor="rgba(0, 0, 0, 0.1)"
        zoomable={true}
        zoomStep={1}       // Match canvas: ~9 clicks min to max (default 10 is too fast)
        pannable={true}
        style={{
          backgroundColor: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: '4px'
        }}
      />
    </ReactFlow>
  );
});

// Main export - wraps with ReactFlowProvider so inner components can use useReactFlow
export const FlowView = observer(({ onNodeClick }: FlowViewProps) => {
  return (
    <ReactFlowProvider>
      <FlowViewInner onNodeClick={onNodeClick} />
    </ReactFlowProvider>
  );
});
