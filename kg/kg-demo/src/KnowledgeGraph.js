import React, { useCallback, useEffect } from 'react';
import ReactFlow, { MiniMap, Controls, useNodesState, useEdgesState, applyNodeChanges, applyEdgeChanges } from 'react-flow-renderer';
import jsonData from './kg_relations.json';

const KnowledgeGraph = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    useEffect(() => {
        const nodesMap = new Map();
        const sourcePositionMap = new Map();
        const targetPositionMap = new Map();
        const numColumns = 5;
        const xSpacing = 200;
        const ySpacing = 150;

        jsonData.forEach((item, index) => {
            const sourceId = `${item.source}_${item.sourcetype}`;
            const targetId = `${item.target}_${item.targettype}`;

            if (!nodesMap.has(sourceId)) {
                let sourceRow = sourcePositionMap.get(item.source);
                if (sourceRow === undefined) {
                    sourceRow = sourcePositionMap.size;
                    sourcePositionMap.set(item.source, sourceRow);
                }
                const xPosition = (sourceRow % numColumns) * xSpacing;
                const yPosition = Math.floor(sourceRow / numColumns) * ySpacing;
                nodesMap.set(sourceId, {
                    id: sourceId,
                    type: "default",
                    className: 'custom-node',
                    data: { label: `${item.source}` },
                    position: { x: xPosition, y: yPosition },
                });
            }

            if (!nodesMap.has(targetId)) {
                let targetRow = targetPositionMap.get(item.target);
                if (targetRow === undefined) {
                    targetRow = targetPositionMap.size;
                    targetPositionMap.set(item.target, targetRow);
                }
                const xPosition = (targetRow % numColumns) * xSpacing;
                const yPosition = Math.floor(targetRow / numColumns) * ySpacing + 300;
                nodesMap.set(targetId, {
                    id: targetId,
                    type: "default",
                    className: 'custom-node',
                    data: { label: `${item.target}` },
                    position: { x: xPosition, y: yPosition },
                    style: { backgroundColor: "#EBE1D2" }
                });
            }

            setEdges(prev => [...prev, {
                id: `e${index}`,
                source: sourceId,
                target: targetId,
                animated: true,
                label: item.relation,
            }]);
        });

        setNodes([...nodesMap.values()]);
    }, []);



    const onNodeDragStop = useCallback((event, node) => {
        setNodes((nds) => nds.map((n) => (n.id === node.id ? node : n)));
    }, [setNodes]);

    return (
        <div style={{ height: '100vh' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeDragStop={onNodeDragStop}
            >
                <MiniMap />
                <Controls />
            </ReactFlow>
        </div>
    );
};

export default KnowledgeGraph;
