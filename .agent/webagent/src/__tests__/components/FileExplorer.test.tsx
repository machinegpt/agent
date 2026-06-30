import { render, screen, fireEvent, act } from "@testing-library/react";
import FileExplorer from "../../components/FileExplorer";
import { TestWrapper } from "../TestWrapper";

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("navigator", { clipboard: { writeText: vi.fn() } });
});

describe("FileExplorer", () => {
  it("shows empty state when no files", () => {
    render(<TestWrapper><FileExplorer files={{}} /></TestWrapper>);
    expect(screen.getByText("NO WORKSPACE FILES")).toBeInTheDocument();
    expect(screen.getByText(/0 files/i)).toBeInTheDocument();
  });

  it("shows file names in sidebar", () => {
    const files = { "JINX.yaml": "state: {}", "plan.json": "[]" };
    render(<TestWrapper><FileExplorer files={files} /></TestWrapper>);
    const jinxEls = screen.getAllByText("JINX.yaml");
    expect(jinxEls.length).toBeGreaterThanOrEqual(1);
    const planEls = screen.getAllByText("plan.json");
    expect(planEls.length).toBeGreaterThanOrEqual(1);
  });

  it("selects first file by default", () => {
    const files = { "state.json": '{"phase":"idle"}' };
    render(<TestWrapper><FileExplorer files={files} /></TestWrapper>);
    const stateEls = screen.getAllByText("state.json");
    expect(stateEls.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/idle/)).toBeInTheDocument();
  });

  it("renders placeholder when no file selected", () => {
    render(<TestWrapper><FileExplorer files={{}} /></TestWrapper>);
    expect(screen.getByText(/SELECT A FILE/i)).toBeInTheDocument();
  });

  it("switches displayed content on file click", () => {
    const files = { "a.yaml": "alpha", "b.yaml": "beta" };
    render(<TestWrapper><FileExplorer files={files} /></TestWrapper>);
    fireEvent.click(screen.getByText("b.yaml"));
    expect(screen.getByText("beta")).toBeInTheDocument();
  });

  it("copies content to clipboard on copy button click", async () => {
    navigator.clipboard.writeText = vi.fn().mockResolvedValue(undefined);

    const files = { "test.yaml": "content-to-copy" };
    render(<TestWrapper><FileExplorer files={files} /></TestWrapper>);
    fireEvent.click(screen.getByText(/Copy to clipboard/i));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("content-to-copy");
  });

  it("shows copy error and clears it after timeout", async () => {
    vi.useFakeTimers();
    navigator.clipboard.writeText = vi.fn().mockRejectedValue(new Error("denied"));

    render(<TestWrapper><FileExplorer files={{ "test.yaml": "content-to-copy" }} /></TestWrapper>);
    fireEvent.click(screen.getByText(/Copy to clipboard/i));

    await vi.advanceTimersByTimeAsync(0);
    expect(screen.getByText(/Copy failed/i)).toBeInTheDocument();

    act(() => { vi.advanceTimersByTime(2000); });
    expect(screen.queryByText(/Copy failed/i)).not.toBeInTheDocument();

    vi.useRealTimers();
  });
});
