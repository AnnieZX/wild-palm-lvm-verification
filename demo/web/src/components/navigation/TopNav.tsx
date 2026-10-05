import { useEffect, useState } from "react";

export const NAV = [
  { id: "overview", label: "Overview" },
  { id: "explorer", label: "Detection Explorer" },
  { id: "context-shift", label: "Context Shift" },
  { id: "scale", label: "Model Scale" },
  { id: "difficulty", label: "Detection Difficulty" },
  { id: "review", label: "Human Review" },
  { id: "semantic", label: "Semantic GT" },
  { id: "method", label: "Method" },
];

export function TopNav() {
  const [active, setActive] = useState("overview");
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -60% 0px" },
    );
    NAV.forEach((n) => {
      const el = document.getElementById(n.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);
  return (
    <nav className="topnav">
      <div className="page topnav-inner">
        <a className="brand" href="#overview" style={{ color: "inherit", textDecoration: "none" }}>
          WILD PALM
        </a>
        <div className="navlinks">
          {NAV.map((n) => (
            <a key={n.id} href={`#${n.id}`} className={active === n.id ? "active" : ""}>
              {n.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
