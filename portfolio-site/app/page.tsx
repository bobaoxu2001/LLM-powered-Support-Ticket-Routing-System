import Image from "next/image";
import type { LucideIcon } from "lucide-react";
import {
  Brain,
  Zap,
  Shield,
  TrendingUp,
  ArrowRight,
  ExternalLink,
  Database,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Target,
  ChevronRight,
  GitBranch,
  Layers,
  DollarSign,
  BookOpen,
  MessageSquare,
  Tag,
  Cpu,
  Scale,
  FlaskConical,
} from "lucide-react";

// ─────────────────────────────────────────────
// Shared primitives
// ─────────────────────────────────────────────

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs font-semibold uppercase tracking-wider">
      {children}
    </span>
  );
}

function SectionHeader({
  badge,
  title,
  subtitle,
}: {
  badge: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="text-center max-w-2xl mx-auto">
      <Badge>{badge}</Badge>
      <h2 className="mt-4 text-3xl md:text-4xl font-bold text-white tracking-tight">
        {title}
      </h2>
      <p className="mt-4 text-slate-400 text-lg leading-relaxed">{subtitle}</p>
    </div>
  );
}

// ─────────────────────────────────────────────
// 1. Hero
// ─────────────────────────────────────────────

const HERO_KPIS: { value: string; label: string; sub: string; icon: LucideIcon }[] = [
  {
    value: "8,325",
    label: "Structured Ticket Rows",
    sub: "Deduplicated Kaggle dataset",
    icon: Database,
  },
  {
    value: "1,665",
    label: "Held-out Benchmark Tickets",
    sub: "Stratified 80/20 split",
    icon: BarChart3,
  },
  {
    value: "+21.1 pts",
    label: "Macro-F1 Lift",
    sub: "Improved ML vs keyword baseline",
    icon: TrendingUp,
  },
  {
    value: "4-Stage",
    label: "Routing Cascade",
    sub: "Rules → ML → LLM → Human",
    icon: Layers,
  },
  {
    value: "Sweep",
    label: "Cost–Coverage Threshold",
    sub: "Policy operating curve",
    icon: Target,
  },
];

function Hero() {
  return (
    <section
      id="hero"
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-6 py-28"
    >
      {/* Background */}
      <div className="absolute inset-0 bg-[#080c1a]" />
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.018)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.018)_1px,transparent_1px)] bg-[size:64px_64px]" />
      <div className="absolute top-0 right-[10%] w-[700px] h-[700px] rounded-full bg-blue-700/[0.07] blur-[120px] pointer-events-none" />
      <div className="absolute bottom-10 left-[5%] w-[500px] h-[500px] rounded-full bg-violet-700/[0.07] blur-[100px] pointer-events-none" />

      <div className="relative z-10 max-w-6xl mx-auto w-full text-center">
        {/* Disclaimer badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300 text-sm font-medium mb-8">
          <Brain className="w-4 h-4 flex-shrink-0" />
          Portfolio Project · Public Kaggle Data · No Production Claims
        </div>

        {/* Title */}
        <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.08] mb-6">
          <span className="bg-gradient-to-r from-blue-400 via-sky-300 to-violet-400 bg-clip-text text-transparent">
            AI-powered Support
          </span>
          <br />
          <span className="text-white">Operations Optimization</span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg md:text-xl text-slate-400 max-w-3xl mx-auto mb-10 leading-relaxed">
          A gDATA-style case routing system that combines rules, calibrated ML,
          LLM fallback, and human triage to make support ticket routing{" "}
          <span className="text-slate-200 font-medium">
            measurable, cost-aware, and controllable.
          </span>
        </p>

        {/* CTAs */}
        <div className="flex flex-wrap items-center justify-center gap-4 mb-16">
          <a
            href="https://github.com/bobaoxu2001/LLM-powered-Support-Ticket-Routing-System"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 text-white font-semibold hover:from-blue-500 hover:to-violet-500 transition-all duration-200 shadow-xl shadow-blue-600/20"
          >
            <ExternalLink className="w-4 h-4" />
            View GitHub
          </a>
          <a
            href="https://github.com/bobaoxu2001/LLM-powered-Support-Ticket-Routing-System/blob/main/README.md"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl bg-white/5 border border-white/10 text-white font-semibold hover:bg-white/10 hover:border-white/20 transition-all duration-200"
          >
            <BookOpen className="w-4 h-4" />
            Read Case Study
          </a>
        </div>

        {/* KPI cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {HERO_KPIS.map((k) => {
            const Icon = k.icon;
            return (
              <div
                key={k.label}
                className="group bg-white/[0.04] border border-white/[0.08] rounded-2xl p-5 text-left hover:bg-white/[0.07] hover:border-white/[0.14] transition-all duration-200"
              >
                <Icon className="w-5 h-5 text-blue-400 mb-3" />
                <div className="text-2xl font-bold text-white leading-none mb-1.5">
                  {k.value}
                </div>
                <div className="text-xs font-medium text-slate-300 mb-1">
                  {k.label}
                </div>
                <div className="text-xs text-slate-500">{k.sub}</div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 2. Business Challenge
// ─────────────────────────────────────────────

function BusinessChallenge() {
  const cards: {
    icon: LucideIcon;
    title: string;
    desc: string;
    accent: string;
    iconBg: string;
  }[] = [
    {
      icon: Layers,
      title: "Reduce Manual Review Load",
      desc: "Manual triage of every incoming ticket is costly and doesn't scale. Automation needs to be safe enough to handle high-confidence cases without burdening agents.",
      accent: "border-blue-500/20",
      iconBg: "bg-blue-500/10 border-blue-500/20 text-blue-400",
    },
    {
      icon: DollarSign,
      title: "Control LLM Invocation Cost",
      desc: "LLM-only routing invokes a language model for every ticket — expensive and hard to cost-control at scale. Smart routing limits LLM calls to truly ambiguous cases.",
      accent: "border-violet-500/20",
      iconBg: "bg-violet-500/10 border-violet-500/20 text-violet-400",
    },
    {
      icon: Shield,
      title: "Preserve Routing Quality & Safety",
      desc: "Misrouted tickets harm customer trust and operations. Uncertain cases and LLM failures must always land in human triage — the system should never fail silently.",
      accent: "border-indigo-500/20",
      iconBg: "bg-indigo-500/10 border-indigo-500/20 text-indigo-400",
    },
  ];

  return (
    <section id="challenge" className="py-24 px-6 bg-[#0a0e1c]">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          badge="Business Context"
          title="The Business Challenge"
          subtitle="Support teams face competing goals: automation coverage, human review load, LLM API cost, and routing quality. A principled system makes those tradeoffs explicit and measurable."
        />
        <div className="mt-16 grid md:grid-cols-3 gap-6">
          {cards.map((c) => {
            const Icon = c.icon;
            return (
              <div
                key={c.title}
                className={`bg-white/[0.04] border ${c.accent} rounded-2xl p-8 hover:bg-white/[0.06] transition-all duration-200`}
              >
                <div
                  className={`inline-flex p-3 rounded-xl border ${c.iconBg} mb-6`}
                >
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-3">
                  {c.title}
                </h3>
                <p className="text-slate-400 leading-relaxed text-sm">
                  {c.desc}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 3. Data-to-Decision Workflow
// ─────────────────────────────────────────────

function WorkflowSection() {
  const steps: {
    num: string;
    title: string;
    desc: string;
    icon: LucideIcon;
  }[] = [
    {
      num: "01",
      title: "Public Support Data",
      desc: "Two Kaggle datasets: Twitter customer messages and a structured ticket dataset with metadata fields.",
      icon: Database,
    },
    {
      num: "02",
      title: "Label Provenance",
      desc: "Ticket Type metadata columns map to issue labels. Labels are never derived from model predictions or keyword rules.",
      icon: Tag,
    },
    {
      num: "03",
      title: "Supervised Benchmark",
      desc: "Stratified 80/20 holdout split. Keyword, ML baseline, and improved ML each evaluated on 1,665 unseen tickets.",
      icon: FlaskConical,
    },
    {
      num: "04",
      title: "Routing Cascade",
      desc: "4-stage system: deterministic rules, calibrated ML, LLM fallback, human triage as the safety net.",
      icon: GitBranch,
    },
    {
      num: "05",
      title: "Threshold Policy",
      desc: "Confidence sweep generates a cost–coverage operating curve to inform threshold selection.",
      icon: Scale,
    },
    {
      num: "06",
      title: "Decision Artifacts",
      desc: "Queue distributions, threshold guides, benchmark metrics, and routing KPIs exported for review.",
      icon: BarChart3,
    },
  ];

  return (
    <section id="workflow" className="py-24 px-6 bg-[#080c1a]">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          badge="Methodology"
          title="From Data to Decision"
          subtitle="A structured pipeline that goes from raw public data to an actionable routing policy with explicit, measurable tradeoffs."
        />

        <div className="mt-16 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-5">
          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <div
                key={step.num}
                className="relative flex flex-col items-center text-center group"
              >
                {/* Connector for large screens */}
                {i < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-7 left-[calc(50%+24px)] right-[-50%] h-px bg-gradient-to-r from-blue-500/30 to-violet-500/10 z-0" />
                )}
                <div className="relative z-10 w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-violet-600/15 border border-blue-500/20 flex items-center justify-center mb-3 group-hover:border-blue-400/40 group-hover:from-blue-600/30 transition-all duration-200">
                  <Icon className="w-6 h-6 text-blue-400" />
                </div>
                <div className="font-mono text-[11px] text-blue-400/50 mb-1">
                  {step.num}
                </div>
                <div className="text-sm font-semibold text-white mb-1.5 leading-tight">
                  {step.title}
                </div>
                <div className="text-xs text-slate-500 leading-relaxed">
                  {step.desc}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 4. Solution Architecture
// ─────────────────────────────────────────────

function SolutionArchitecture() {
  const stages: {
    num: string;
    title: string;
    desc: string;
    icon: LucideIcon;
    gradient: string;
    border: string;
    text: string;
    badge: string;
  }[] = [
    {
      num: "1",
      title: "Rule-based Routing",
      desc: "Deterministic pattern matching for obvious cases — instant, zero model overhead, highest reliability.",
      icon: Zap,
      gradient: "from-yellow-600/15 to-amber-600/10",
      border: "border-yellow-500/20",
      text: "text-yellow-300",
      badge: "bg-yellow-500/10 border-yellow-500/20 text-yellow-400",
    },
    {
      num: "2",
      title: "Calibrated ML",
      desc: "TF-IDF + Logistic Regression with isotonic calibration. Auto-routes when confidence ≥ high threshold.",
      icon: Cpu,
      gradient: "from-blue-600/15 to-blue-700/10",
      border: "border-blue-500/20",
      text: "text-blue-300",
      badge: "bg-blue-500/10 border-blue-500/20 text-blue-400",
    },
    {
      num: "3",
      title: "LLM Fallback",
      desc: "LLM performs issue-type classification for ambiguous low-confidence tickets. Not invoked for every ticket.",
      icon: Brain,
      gradient: "from-violet-600/15 to-purple-700/10",
      border: "border-violet-500/20",
      text: "text-violet-300",
      badge: "bg-violet-500/10 border-violet-500/20 text-violet-400",
    },
    {
      num: "4",
      title: "Human Triage",
      desc: "Middle-confidence ambiguity and LLM failures always route here. The safety net — never fails silently.",
      icon: Shield,
      gradient: "from-emerald-600/15 to-green-700/10",
      border: "border-emerald-500/20",
      text: "text-emerald-300",
      badge: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
    },
  ];

  return (
    <section id="architecture" className="py-24 px-6 bg-[#0a0e1c]">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          badge="System Design"
          title="4-Stage Routing Cascade"
          subtitle="Each stage handles the cases it's best suited for. Ambiguous tickets pass downstream; uncertain cases always reach a human."
        />

        {/* Flow diagram */}
        <div className="mt-14 flex flex-wrap items-center justify-center gap-2 md:gap-3 mb-10">
          <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-slate-300 text-sm font-medium">
            <MessageSquare className="w-4 h-4 text-slate-400" />
            Incoming Ticket
          </div>
          {stages.map((s) => (
            <div key={s.num} className="flex items-center gap-2 md:gap-3">
              <ChevronRight className="w-4 h-4 text-slate-600 flex-shrink-0" />
              <div
                className={`px-4 py-2.5 rounded-xl bg-gradient-to-r ${s.gradient} border ${s.border} text-sm font-semibold ${s.text}`}
              >
                {s.title}
              </div>
            </div>
          ))}
        </div>

        {/* Stage cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
          {stages.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.num}
                className={`bg-white/[0.04] border ${s.border} rounded-2xl p-6 hover:bg-white/[0.06] transition-all duration-200`}
              >
                <div
                  className={`inline-flex p-2.5 rounded-xl border ${s.badge} mb-4`}
                >
                  <Icon className="w-5 h-5" />
                </div>
                <div className="text-xs font-mono text-slate-500 mb-1">
                  Stage {s.num}
                </div>
                <h3 className="text-base font-semibold text-white mb-2.5">
                  {s.title}
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {s.desc}
                </p>
              </div>
            );
          })}
        </div>

        {/* LLM note */}
        <div className="mt-7 flex items-start gap-3 p-5 rounded-2xl bg-violet-500/5 border border-violet-500/15">
          <Brain className="w-5 h-5 text-violet-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-slate-300 leading-relaxed">
            <span className="font-semibold text-violet-300">
              Not an LLM-only design.
            </span>{" "}
            LLM calls are intentionally limited to low-confidence ambiguous
            cases to control API cost and improve operational safety. The
            majority of tickets are handled by rules or ML without any LLM
            invocation.
          </p>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 5. Results & Insights
// ─────────────────────────────────────────────

function ResultsInsights() {
  const cards: {
    title: string;
    file: string;
    tag: string;
    tagStyle: string;
    desc: string;
    note: string;
    noteIcon: LucideIcon;
  }[] = [
    {
      title: "Operations Overview",
      file: "/images/dashboard_overview.png",
      tag: "Proxy + Estimated Metrics",
      tagStyle:
        "bg-amber-500/10 border border-amber-500/20 text-amber-300",
      desc: "Shows routing-stage distribution, human triage rate, LLM invocation rate, average confidence score, estimated cost per ticket, and queue distribution across all processed tickets.",
      note: "Human triage rate is a routing-system proxy, not a downstream escalation rate. Cost estimates are analytic, not measured from live LLM calls.",
      noteIcon: AlertCircle,
    },
    {
      title: "Cost–Coverage Policy Tradeoff",
      file: "/images/policy_tradeoff.png",
      tag: "Estimated Analytic Metrics",
      tagStyle: "bg-blue-500/10 border border-blue-500/20 text-blue-300",
      desc: "Shows how confidence thresholds shift tickets between auto-routing, LLM fallback, and human review. Thresholds are operating policy choices — not just model parameters — and must be selected with risk tolerance and LLM budget in mind.",
      note: "Computed analytically from ML confidence scores. No live LLM calls required for this sweep.",
      noteIcon: AlertCircle,
    },
    {
      title: "Model Evaluation",
      file: "/images/model_evaluation.png",
      tag: "Measured on Holdout Data",
      tagStyle:
        "bg-emerald-500/10 border border-emerald-500/20 text-emerald-300",
      desc: "Measured ML vs keyword baseline on 1,665 held-out metadata-derived tickets. Improved ML raises Macro-F1 from 12.1% to 33.2% vs keyword rules. ML baseline's 59.4% accuracy reflects majority-class bias; the improved model trades raw accuracy for better class balance. Limitations are transparent.",
      note: "Labels are metadata-derived (Ticket Type field), not human-reviewed. Absolute scores remain modest and should be interpreted cautiously.",
      noteIcon: AlertCircle,
    },
  ];

  return (
    <section id="results" className="py-24 px-6 bg-[#080c1a]">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          badge="Results"
          title="Results & Insights"
          subtitle="Three views of the system: operational health, policy tradeoffs, and measured model quality. Metric types are clearly distinguished."
        />

        <div className="mt-16 space-y-20">
          {cards.map((card, i) => {
            const NoteIcon = card.noteIcon;
            const isEven = i % 2 === 0;
            return (
              <div
                key={card.title}
                className="grid lg:grid-cols-2 gap-10 items-center"
              >
                {/* Image side */}
                <div
                  className={`${isEven ? "lg:order-1" : "lg:order-2"}`}
                >
                  <div className="rounded-2xl overflow-hidden border border-white/[0.08] bg-slate-900/60 shadow-2xl shadow-black/40">
                    <Image
                      src={card.file}
                      alt={card.title}
                      width={1200}
                      height={750}
                      className="w-full h-auto"
                      priority={i === 0}
                    />
                  </div>
                </div>

                {/* Text side */}
                <div className={`${isEven ? "lg:order-2" : "lg:order-1"}`}>
                  <span
                    className={`inline-flex px-3 py-1 rounded-full text-xs font-semibold mb-4 ${card.tagStyle}`}
                  >
                    {card.tag}
                  </span>
                  <h3 className="text-2xl md:text-3xl font-bold text-white mb-4 tracking-tight">
                    {card.title}
                  </h3>
                  <p className="text-slate-300 leading-relaxed mb-5 text-[15px]">
                    {card.desc}
                  </p>
                  <div className="flex items-start gap-2.5 p-4 rounded-xl bg-white/[0.04] border border-white/[0.08]">
                    <NoteIcon className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {card.note}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 6. Executive Recommendation
// ─────────────────────────────────────────────

function ExecutiveRecommendation() {
  const bullets: { icon: LucideIcon; text: string }[] = [
    {
      icon: Zap,
      text: "Keep deterministic rules for high-signal obvious cases — no model overhead, highest reliability.",
    },
    {
      icon: Cpu,
      text: "Use calibrated ML for high-confidence routing — cost-efficient, fast, and measurable.",
    },
    {
      icon: Brain,
      text: "Trigger LLM classification only for ambiguous low-confidence tickets — not for every case.",
    },
    {
      icon: Shield,
      text: "Route uncertain cases and LLM failures to human triage — never fail silently.",
    },
    {
      icon: Scale,
      text: "Select confidence thresholds based on risk tolerance, review capacity, and LLM budget.",
    },
  ];

  return (
    <section id="recommendation" className="py-24 px-6 bg-[#0a0e1c]">
      <div className="max-w-5xl mx-auto">
        <div className="relative rounded-3xl overflow-hidden border border-blue-500/15">
          {/* Panel glow bg */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-950/60 via-[#0d1428] to-violet-950/60" />
          <div className="absolute top-0 right-0 w-80 h-80 bg-blue-600/[0.08] rounded-full blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-60 h-60 bg-violet-600/[0.08] rounded-full blur-3xl pointer-events-none" />

          <div className="relative z-10 p-10 md:p-14">
            <div className="flex flex-wrap items-center gap-3 mb-6">
              <div className="p-2.5 rounded-xl bg-blue-500/15 border border-blue-500/25">
                <Target className="w-6 h-6 text-blue-400" />
              </div>
              <Badge>Business Recommendation</Badge>
            </div>

            <h2 className="text-3xl md:text-4xl font-bold text-white mb-2 tracking-tight">
              Use a hybrid routing policy
            </h2>
            <p className="text-slate-400 text-lg mb-10">
              rather than LLM-only or rules-only routing.
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              {bullets.map((b, i) => {
                const Icon = b.icon;
                return (
                  <div
                    key={i}
                    className="flex items-start gap-3 p-4 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.07] transition-colors"
                  >
                    <div className="p-1.5 rounded-lg bg-blue-500/15 border border-blue-500/20 flex-shrink-0 mt-0.5">
                      <Icon className="w-4 h-4 text-blue-400" />
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      {b.text}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 7. Limitations & Next Steps
// ─────────────────────────────────────────────

function LimitationsNextSteps() {
  const limitations = [
    "Metadata-derived labels are not human-reviewed production gold labels",
    "Kaggle ticket descriptions are templated and may not fully reflect production support traffic",
    "The supervised benchmark covers 3 of 6 routing classes reliably present in the raw Kaggle metadata",
    "Absolute ML scores are modest; interpret cautiously against real-world benchmarks",
  ];

  const next = [
    "Build a 500–2,000 row human-reviewed eval set for more reliable per-class F1 signal",
    "Add queue-level SLA policies and risk-tier threshold settings per support category",
    "Track true downstream escalation and resolution outcomes beyond routing decisions",
    "Add calibration plots, reliability monitoring, and model drift detection",
  ];

  return (
    <section id="limitations" className="py-24 px-6 bg-[#080c1a]">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          badge="Honest Assessment"
          title="Limitations & Next Steps"
          subtitle="Transparent caveats are a mark of rigorous data science. Here is what this system does and does not claim."
        />

        <div className="mt-16 grid md:grid-cols-2 gap-8">
          <div className="bg-white/[0.04] border border-amber-500/15 rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-7">
              <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <AlertCircle className="w-5 h-5 text-amber-400" />
              </div>
              <h3 className="text-lg font-semibold text-white">
                Current Limitations
              </h3>
            </div>
            <ul className="space-y-4">
              {limitations.map((l, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 text-sm text-slate-400 leading-relaxed"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-400/50 flex-shrink-0 mt-[7px]" />
                  {l}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-white/[0.04] border border-blue-500/15 rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-7">
              <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <ArrowRight className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="text-lg font-semibold text-white">Next Steps</h3>
            </div>
            <ul className="space-y-4">
              {next.map((s, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 text-sm text-slate-400 leading-relaxed"
                >
                  <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                  {s}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 8. Tech Stack
// ─────────────────────────────────────────────

function TechStack() {
  const categories: { label: string; color: string; items: string[] }[] = [
    {
      label: "Data & ML",
      color: "text-blue-400",
      items: ["Python", "pandas", "scikit-learn", "Logistic Regression", "TF-IDF"],
    },
    {
      label: "NLP",
      color: "text-violet-400",
      items: ["Tokenization", "Char n-grams", "Text normalization", "FeatureUnion"],
    },
    {
      label: "AI & LLM",
      color: "text-purple-400",
      items: ["OpenAI API", "GPT-4.1-mini", "JSON parsing", "LLM fallback"],
    },
    {
      label: "Data Sources",
      color: "text-sky-400",
      items: ["Kaggle", "Twitter support", "Ticket dataset", "Metadata labels"],
    },
    {
      label: "App & Viz",
      color: "text-emerald-400",
      items: ["Streamlit", "Matplotlib", "seaborn", "Next.js", "Tailwind"],
    },
  ];

  return (
    <section id="stack" className="py-24 px-6 bg-[#0a0e1c]">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          badge="Implementation"
          title="Tech Stack"
          subtitle="Built entirely with open-source tools on public Kaggle data. No proprietary data sources."
        />

        <div className="mt-16 grid grid-cols-2 md:grid-cols-5 gap-5">
          {categories.map((cat) => (
            <div
              key={cat.label}
              className="bg-white/[0.04] border border-white/[0.08] rounded-2xl p-5"
            >
              <div
                className={`text-xs font-bold uppercase tracking-widest mb-4 ${cat.color}`}
              >
                {cat.label}
              </div>
              <div className="flex flex-col gap-2">
                {cat.items.map((item) => (
                  <div
                    key={item}
                    className="px-3 py-1.5 rounded-lg bg-white/[0.05] border border-white/[0.07] text-xs text-slate-300 text-center leading-snug"
                  >
                    {item}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────
// 9. Footer
// ─────────────────────────────────────────────

function SiteFooter() {
  return (
    <footer className="py-14 px-6 bg-[#080c1a] border-t border-white/[0.06]">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <div className="text-base font-semibold text-white mb-1">
            Allen Xu
          </div>
          <div className="text-sm text-slate-500">
            Portfolio project · Public Kaggle data · No proprietary data · No
            production claims
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <a
            href="https://github.com/bobaoxu2001/LLM-powered-Support-Ticket-Routing-System"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 text-white text-sm font-semibold hover:from-blue-500 hover:to-violet-500 transition-all duration-200"
          >
            <ExternalLink className="w-4 h-4" />
            GitHub Repository
          </a>
          <a
            href="https://github.com/bobaoxu2001/LLM-powered-Support-Ticket-Routing-System/blob/main/README.md"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm font-semibold hover:bg-white/10 transition-all duration-200"
          >
            <BookOpen className="w-4 h-4" />
            README / Case Study
          </a>
        </div>
      </div>
    </footer>
  );
}

// ─────────────────────────────────────────────
// Page
// ─────────────────────────────────────────────

export default function Home() {
  return (
    <main className="min-h-screen bg-[#080c1a] text-white overflow-x-hidden">
      <Hero />
      <BusinessChallenge />
      <WorkflowSection />
      <SolutionArchitecture />
      <ResultsInsights />
      <ExecutiveRecommendation />
      <LimitationsNextSteps />
      <TechStack />
      <SiteFooter />
    </main>
  );
}
