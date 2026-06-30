import { render, screen, fireEvent } from "@testing-library/react";
import FileExplorer from "../../components/FileExplorer";
import { TestWrapper } from "../TestWrapper";

beforeEach(() => {
  localStorage.clear();
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
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    const files = { "test.yaml": "content-to-copy" };
    render(<TestWrapper><FileExplorer files={files} /></TestWrapper>);
    fireEvent.click(screen.getByText(/Copy to clipboard/i));
    expect(writeText).toHaveBeenCalledWith("content-to-copy");
  });
});
