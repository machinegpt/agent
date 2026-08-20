import { render, screen } from "@testing-library/react";
import CognitiveLoop from "../../components/CognitiveLoop";
import { TestWrapper } from "../TestWrapper";

describe("CognitiveLoop", () => {
  it("renders title and status", () => {
    render(<TestWrapper><CognitiveLoop currentStatus="idle" /></TestWrapper>);
    expect(screen.getByText("Idle")).toBeInTheDocument();
  });

  it("renders all 7 phases on desktop", () => {
    render(<TestWrapper><CognitiveLoop currentStatus="idle" /></TestWrapper>);
    const labels = ["Perceive", "Analyze", "Plan", "Execute", "Verify", "Commit", "Completed"];
    for (const label of labels) {
      const els = screen.getAllByText(label);
      expect(els.length).toBeGreaterThanOrEqual(1);
    }
  });

  it("highlights active phase", () => {
    render(<TestWrapper><CognitiveLoop currentStatus="execute" /></TestWrapper>);
    const executeEls = screen.getAllByText("Execute");
    expect(executeEls.length).toBeGreaterThanOrEqual(1);
    const activeBadges = screen.getAllByText("ACTIVE");
    expect(activeBadges).toHaveLength(1);
  });

  it("shows error badge on error status", () => {
    render(<TestWrapper><CognitiveLoop currentStatus="error" /></TestWrapper>);
    expect(screen.getByText("LOOP EXCEPTION DETECTED")).toBeInTheDocument();
    const errorBadges = screen.getAllByText("ERROR");
    expect(errorBadges.length).toBeGreaterThanOrEqual(1);
  });

  it("shows idle message waiting for agent", () => {
    render(<TestWrapper><CognitiveLoop currentStatus="idle" /></TestWrapper>);
    const idleMsgs = screen.getAllByText(/Waiting for JINX Agent/i);
    expect(idleMsgs.length).toBeGreaterThanOrEqual(1);
  });

  it("renders completed status", () => {
    render(<TestWrapper><CognitiveLoop currentStatus="completed" /></TestWrapper>);
    const completedEls = screen.getAllByText("Completed");
    expect(completedEls.length).toBeGreaterThanOrEqual(1);
  });
});
