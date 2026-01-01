import { Node, Edge, Position } from 'reactflow';
import { hierarchy, HierarchyNode } from 'd3-hierarchy';
import { FlowNodeData } from '@/app/stores/sessionStore';

// Define fallback dimensions for the initial render pass.
// These are used before the actual node sizes are measured.
// Updated to realistic heights that account for header + content + tools + footer.
export const NODE_FALLBACK_DIMENSIONS = {
  turn: { width: 380, height: 320 }, // Base: header(40) + content(L=160) + tools(80) + footer(20) + padding. Width increased for long text.
  principal: { width: 380, height: 280 }, // Principal node - matches turn width
  agent: { width: 360, height: 260 }, // Agent node
  default: { width: 320, height: 150 },
  gather: { width: 380, height: 35 }, // Gather node aligned with other cards' width
  epoch_separator: { width: 380, height: 40 }, // Epoch separator - full width, minimal height
};

// 6 fixed heights for the content box (including padding)
export const CONTENT_BOX_HEIGHTS = {
  XS: 0, // height is 0 for no content
  S: 24 + 16, // h-6 + padding
  M: 80 + 16, // h-20 + padding
  L: 144 + 16, // h-36 + padding
  XL: 208 + 16, // h-52 + padding
  XXL: 320 + 16, // h-80 + padding (capped for very long content)
};

// Calculate content box height level based on content length
const getContentHeightLevel = (content: string): 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL' => {
  if (!content) return 'XS'; // XS - no content

  const length = content.length;
  if (length <= 50) return 'S'; // S - 1 line of text
  if (length <= 200) return 'M'; // M - 5 lines of text
  if (length <= 400) return 'L'; // L - 9 lines of text
  if (length <= 800) return 'XL'; // XL - 13 lines of text
  return 'XXL'; // XXL - 26 lines of text (very long content)
};

// Get the larger content height level
const getMaxContentLevel = (level1: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL', level2: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL'): 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL' => {
  const levels = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];
  const index1 = levels.indexOf(level1);
  const index2 = levels.indexOf(level2);
  return levels[Math.max(index1, index2)] as 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL';
};

// Get the node's actual display content (including streaming content)
const getNodeDisplayContent = (nodeData: FlowNodeData, streamingContentMap: Map<string, string>): string => {
  if (nodeData.content_stream_id && streamingContentMap.has(nodeData.content_stream_id)) {
    return streamingContentMap.get(nodeData.content_stream_id) || '';
  }
  return nodeData.final_content || '';
};

// Type for the data structure used by d3-hierarchy
type HierarchyDatum = Node<FlowNodeData> & {
  children: HierarchyDatum[];
};

function getNodeSize(
  node: HierarchyNode<HierarchyDatum>,
  nodeSizes: Map<string, { width: number; height: number }>,
): { width: number; height: number } {
  const nodeId = node.data.id;
  const measuredSize = nodeSizes.get(nodeId);
  if (measuredSize && measuredSize.width > 0 && measuredSize.height > 0) {
    // Ensure minimum dimensions even for measured sizes
    return {
      width: Math.max(measuredSize.width, 340),
      height: Math.max(measuredSize.height, 150)
    };
  }
  const nodeType = (node.data.data as FlowNodeData)?.nodeType || 'turn';
  return (
    NODE_FALLBACK_DIMENSIONS[
      nodeType as keyof typeof NODE_FALLBACK_DIMENSIONS
    ] || NODE_FALLBACK_DIMENSIONS.default
  );
}

// The new layout function that supports variable node sizes.
export const getLayoutedElements = (
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  nodeSizes: Map<string, { width: number; height: number }>,
  streamingContentMap: Map<string, string> = new Map()
) => {
  if (nodes.length === 0) {
    return { nodes: [], edges: [] };
  }

  const hierarchyNodes: Node<FlowNodeData>[] = [...nodes];
  let hierarchyEdges = [...edges];

  // Handle multiple parent nodes by creating a gather point for any node with more than one parent
  const parentMap = new Map<string, string[]>();
  hierarchyEdges.forEach(edge => {
    if (!parentMap.has(edge.target)) {
      parentMap.set(edge.target, []);
    }
    parentMap.get(edge.target)!.push(edge.source);
  });

  let dummyNodeCounter = 0;
  parentMap.forEach((sources, targetId) => {
    if (sources.length > 1) {
      console.log(`🔧 Creating dummy node for target ${targetId} with ${sources.length} sources:`, sources);

      const dummyNodeId = `dummy-${targetId}-${dummyNodeCounter++}`;
      const dummyNode: Node<FlowNodeData> = {
        id: dummyNodeId,
        type: 'custom',
        data: {
          label: 'Gathering Point',
          nodeType: 'gather',
          status: 'idle'
        },
        position: { x: 0, y: 0 },
      };
      hierarchyNodes.push(dummyNode);

      // Reroute edges to the dummy node
      hierarchyEdges = hierarchyEdges.filter(edge => edge.target !== targetId);
      sources.forEach(sourceId => {
        hierarchyEdges.push({
          id: `${sourceId}->${dummyNodeId}`,
          source: sourceId,
          target: dummyNodeId,
          type: 'custom',
          animated: false
        });
      });
      hierarchyEdges.push({
        id: `${dummyNodeId}->${targetId}`,
        source: dummyNodeId,
        target: targetId,
        type: 'custom',
        animated: false
      });
    }
  });

  // Build the hierarchy structure for d3
  const nodeMap = new Map(hierarchyNodes.map(n => [n.id, { ...n, children: [] as HierarchyDatum[] }]));
  const childIds = new Set<string>();

  hierarchyEdges.forEach(edge => {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);
    if (sourceNode && targetNode) {
      sourceNode.children.push(targetNode);
      childIds.add(edge.target);
    }
  });

  const rootNodes = Array.from(nodeMap.values()).filter(n => !childIds.has(n.id));

  // d3-hierarchy expects a single root object that conforms to the tree structure
  const hierarchyRoot: HierarchyDatum = {
    id: 'root',
    type: 'custom',
    position: { x: 0, y: 0 }, // Dummy position for the root
    data: {
      label: 'Root',
      nodeType: 'turn' as const,
      status: 'idle' as const
    },
    children: rootNodes,
  };

  // Use backend-provided depth instead of d3-hierarchy calculation
  // Create depth-based grouping using backend depth values
  const nodesByDepth = new Map<number, HierarchyNode<HierarchyDatum>[]>();

  // Group nodes by their backend-provided depth
  hierarchyNodes.forEach((node) => {
    const backendDepth = node.data.depth;

    // If backend doesn't provide depth, fall back to calculated depth
    let actualDepth: number;
    if (typeof backendDepth === 'number' && backendDepth > 0) {
      actualDepth = backendDepth;
    } else {
      // Fallback: use d3-hierarchy calculation
      const root = hierarchy(hierarchyRoot);
      let fallbackDepth = 1; // Default depth
      root.each((hierarchyNode) => {
        if (hierarchyNode.data.id === node.id && hierarchyNode.depth > 0) {
          fallbackDepth = hierarchyNode.depth;
        }
      });
      actualDepth = fallbackDepth;
      console.warn(`Node ${node.id} missing backend depth, using fallback: ${actualDepth}`);
    }

    if (!nodesByDepth.has(actualDepth)) {
      nodesByDepth.set(actualDepth, []);
    }

    // Create a mock hierarchy node for compatibility with existing layout code
    const mockHierarchyNode: HierarchyNode<HierarchyDatum> = {
      data: {
        ...node,
        children: []
      },
      depth: actualDepth,
      height: 0,
      parent: null,
      children: [],
      value: undefined,
      x: 0,
      y: 0
    } as unknown as HierarchyNode<HierarchyDatum>;

    nodesByDepth.get(actualDepth)!.push(mockHierarchyNode);
  });

  // Layout configuration
  const LEVEL_SPACING = 80; // Gap between levels (in pixels) - increased for better separation
  const MIN_NODE_HEIGHT = 200; // Minimum node height for consistent spacing - accounts for content boxes
  const VIEWPORT_CENTER_X = 500; // Center X coordinate for viewport

  const levelYs = new Map<number, number>();
  let currentY = 0;

  const finalSortedDepths = Array.from(nodesByDepth.keys()).sort((a, b) => a - b);

  for (const depth of finalSortedDepths) {
    // Set the TOP Y coordinate for this level
    levelYs.set(depth, currentY);

    const nodesOnLevel = nodesByDepth.get(depth) || [];
    let maxLevelHeight = MIN_NODE_HEIGHT; // Start with minimum height

    // Find the tallest node on this level
    nodesOnLevel.forEach(node => {
      const { height } = getNodeSize(node, nodeSizes);
      if (height > maxLevelHeight) {
        maxLevelHeight = height;
      }
    });

    // Next level Y = current level Y + current level height + spacing
    currentY += maxLevelHeight + LEVEL_SPACING;
  }

  // 3. Position nodes within each level with improved horizontal distribution
  // Calculate max content level for each layer
  const layerMaxContentLevels = new Map<number, 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL'>();

  nodesByDepth.forEach((nodesOnLevel, depth) => {
    let maxContentLevel: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL' = 'XS';

    nodesOnLevel.forEach(node => {
      // Only calculate for turn nodes that have content
      if (node.data.data?.nodeType === 'turn') {
        // Get actual display content (including streaming content)
        const displayContent = getNodeDisplayContent(node.data.data, streamingContentMap);
        const contentLevel = getContentHeightLevel(displayContent);
        maxContentLevel = getMaxContentLevel(maxContentLevel, contentLevel);
      }
    });

    layerMaxContentLevels.set(depth, maxContentLevel);
    console.log(`📏 Layer ${depth} max content level: ${maxContentLevel}`);
  });

  // First, discover all unique agents and determine the maximum number of concurrent swim lanes
  const allAgentIds = new Set<string>();
  let maxConcurrentAgents = 1;

  nodesByDepth.forEach((nodesOnLevel) => {
    const agentsOnLevel = new Set<string>();
    nodesOnLevel.forEach(node => {
      const agentId = node.data.data?.agent_id;
      if (agentId) {
        allAgentIds.add(agentId);
        agentsOnLevel.add(agentId);
      }
    });
    maxConcurrentAgents = Math.max(maxConcurrentAgents, agentsOnLevel.size);
  });

  console.log(`🏊 Swim lanes: ${allAgentIds.size} unique agents, max ${maxConcurrentAgents} concurrent`);

  // Establish fixed swim lane positions based on maximum concurrent agents
  // This ensures columns don't shift when new agents appear
  const agentColumnPositions = new Map<string, number>(); // agent_id -> x position (center of column)
  const SWIM_LANE_WIDTH = 420; // Width of each swim lane (node width + padding)

  // Position swim lanes based on first occurrence order, but with fixed widths
  const agentOrder: string[] = []; // Track order agents first appear

  nodesByDepth.forEach((nodesOnLevel, depth) => {
    nodesOnLevel.forEach(node => {
      const agentId = node.data.data?.agent_id;
      if (agentId && !agentOrder.includes(agentId)) {
        agentOrder.push(agentId);
      }
    });
  });

  // Calculate swim lane center positions
  const totalSwimLaneWidth = agentOrder.length * SWIM_LANE_WIDTH;
  const swimLaneStartX = VIEWPORT_CENTER_X - (totalSwimLaneWidth / 2) + (SWIM_LANE_WIDTH / 2);

  agentOrder.forEach((agentId, index) => {
    const laneCenter = swimLaneStartX + (index * SWIM_LANE_WIDTH);
    agentColumnPositions.set(agentId, laneCenter);
    console.log(`🏊 Swim lane ${index}: agent ${agentId} at x=${laneCenter}`);
  });

  // Now position nodes using their swim lane positions
  nodesByDepth.forEach((nodesOnLevel, depth) => {
    const levelY = levelYs.get(depth) || 0;

    nodesOnLevel.forEach((node) => {
      const { width } = getNodeSize(node, nodeSizes);
      const agentId = node.data.data?.agent_id;

      let x: number;
      if (agentId && agentColumnPositions.has(agentId)) {
        // Use the swim lane center, then offset to position left edge
        const laneCenter = agentColumnPositions.get(agentId)!;
        x = laneCenter - (width / 2);
      } else {
        // No agent ID - center the node
        x = VIEWPORT_CENTER_X - (width / 2);
      }

      node.data.position = { x, y: levelY };
      console.log(`📍 Positioned ${node.data.data?.nodeType} node '${node.data.data?.label}' at (${x}, ${levelY}), agent: ${agentId || 'none'}`);
    });
  });

  // 4. Apply the calculated positions to the original nodes
  const finalNodes = nodes.map((node) => {
    // Find this node in the hierarchy to get its calculated position
    let calculatedPosition = { x: 0, y: 0 };
    let actualDepth = node.data.depth || 0; // Use backend depth directly
    let layerMaxContentLevel: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL' = 'XS';

    for (const [depth, nodesOnLevel] of nodesByDepth) {
      const hierarchyNode = nodesOnLevel.find(h => h.data.id === node.id);
      if (hierarchyNode) {
        calculatedPosition = hierarchyNode.data.position;
        actualDepth = depth; // Use the depth from our grouping
        layerMaxContentLevel = layerMaxContentLevels.get(depth) || 'XS';
        break;
      }
    }

    return {
      ...node,
      // Position uses top-left coordinates for React Flow
      position: calculatedPosition,
      targetPosition: Position.Top,
      sourcePosition: Position.Bottom,
      // Add hierarchy information for debugging - now using backend depth
      data: {
        ...node.data,
        debugRowNumber: actualDepth, // Use actual depth as row number for debugging
        debugIsHierarchyPlaced: actualDepth > 0,
        layerMaxContentLevel: layerMaxContentLevel, // Add layer max content level
      },
    };
  });

  // 5. Return final nodes and processed edges with custom type
  const finalEdges = edges.map(edge => ({
    ...edge,
    type: 'custom'
  }));

  return { nodes: finalNodes, edges: finalEdges };
};
