"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import type { SelectionStateActions, ViewerSelectionState } from "@/components/viewer/types";

interface SelectionStateContextValue extends ViewerSelectionState, SelectionStateActions {}

const SelectionStateContext = createContext<SelectionStateContextValue | null>(null);

export interface SelectionStateProviderProps {
  children: ReactNode;
  /** Optional initial selection (future: sync with URL or info panel). */
  initialSelectedId?: string | null;
}

/**
 * Tracks selected and hovered bounding-box ids for the orthomosaic viewer.
 */
export function SelectionStateProvider({
  children,
  initialSelectedId = null,
}: SelectionStateProviderProps) {
  const [selectedId, setSelectedId] = useState<string | null>(initialSelectedId);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const select = useCallback((id: string | null) => {
    setSelectedId(id);
  }, []);

  const hover = useCallback((id: string | null) => {
    setHoveredId(id);
  }, []);

  const clear = useCallback(() => {
    setSelectedId(null);
    setHoveredId(null);
  }, []);

  const value = useMemo<SelectionStateContextValue>(
    () => ({
      selectedId,
      hoveredId,
      select,
      hover,
      clear,
    }),
    [selectedId, hoveredId, select, hover, clear],
  );

  return (
    <SelectionStateContext.Provider value={value}>{children}</SelectionStateContext.Provider>
  );
}

/** @alias SelectionStateProvider */
export const SelectionState = SelectionStateProvider;

export function useSelectionState(): SelectionStateContextValue {
  const context = useContext(SelectionStateContext);
  if (!context) {
    throw new Error("useSelectionState must be used within SelectionStateProvider");
  }
  return context;
}
