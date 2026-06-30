import { render, screen } from "@testing-library/react";
import RunSummary from "../../components/RunSummary";
import { TestWrapper } from "../TestWrapper";
import type { AgentSession } from "../../types";

const baseSession: AgentSession = {
  id: "test-1",
  name: "Test Run",
  timestamp: new Date().toISOString(),
  status: "completed",
  elapsedTime: 120,
  stats: { promptTokens: 0, completionTokens: 0, estimatedCost: 0, pid: 1234, hostname: "test-box", os: "linux" },
  plan: [],
  thoughts: [],
  rpcLog: [],
  terminalLog: [],
  diffs: [],
  files: {},
};

describe("RunSummary", () => {
  it("renders elapsed time", () => {
    render(<TestWrapper><RunSummary session={baseSession} /></TestWrapper>);
    expect(screen.getByText("120s")).toBeInTheDocument();
  });

  it("renders PID", () => {
    render(<TestWrapper><RunSummary session={baseSession} /></TestWrapper>);
    expect(screen.getByText("1234")).toBeInTheDocument();
  });

  it("renders hostname", () => {
    render(<TestWrapper><RunSummary session={baseSession} /></TestWrapper>);
    expect(screen.getByText("test-box")).toBeInTheDocument();
  });

  it("shows no-plan message when plan is empty", () => {
    render(<TestWrapper><RunSummary session={baseSession} /></TestWrapper>);
    expect(screen.getByText(/No plan defined yet/i)).toBeInTheDocument();
  });

  it("renders plan steps", () => {
    const session: AgentSession = {
      ...baseSession,
      plan: [
        { id: "s1", title: "Step 1", description: "First step", status: "completed" },
        { id: "s2", title: "Step 2", description: "Second step", status: "running" },
      ],
    };
    render(<TestWrapper><RunSummary session={session} /></TestWrapper>);
    expect(screen.getByText("Step 1")).toBeInTheDocument();
    expect(screen.getByText("Step 2")).toBeInTheDocument();
    expect(screen.getByText("First step")).toBeInTheDocument();
    expect(screen.getByText("Second step")).toBeInTheDocument();
  });

  it("renders scope facts", () => {
    const session: AgentSession = {
      ...baseSession,
      facts: ["Python 3.11", "Uses FastAPI"],
    };
    render(<TestWrapper><RunSummary session={session} /></TestWrapper>);
    expect(screen.getByText("Python 3.11")).toBeInTheDocument();
    expect(screen.getByText("Uses FastAPI")).toBeInTheDocument();
  });

  it("renders open requirements", () => {
    const session: AgentSession = {
      ...baseSession,
      open: ["Add tests", "Fix lint"],
    };
    render(<TestWrapper><RunSummary session={session} /></TestWrapper>);
    expect(screen.getByText("Add tests")).toBeInTheDocument();
    expect(screen.getByText("Fix lint")).toBeInTheDocument();
  });

  it("renders debt items", () => {
    const session: AgentSession = {
      ...baseSession,
      debt: ["Refactor module X"],
    };
    render(<TestWrapper><RunSummary session={session} /></TestWrapper>);
    expect(screen.getByText("Refactor module X")).toBeInTheDocument();
  });

  it("handles N/A pid gracefully", () => {
    const session: AgentSession = {
      ...baseSession,
      stats: { ...baseSession.stats, pid: 0 },
    };
    render(<TestWrapper><RunSummary session={session} /></TestWrapper>);
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });
});
