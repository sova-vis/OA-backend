import { Router, Request, Response } from "express";
import { supabase } from "./lib/supabase";

const router = Router();

/**
 * GET /content/search
 * Search for papers by subject, year, file type
 * Returns structured metadata for navigation
 */
router.get("/search", async (req: Request, res: Response) => {
  try {
    const { subject, year, file_type } = req.query;

    let query = supabase
      .from("papers")
      .select(`
        id,
        year,
        session,
        paper,
        subject_id,
        subjects(code, level),
        paper_files(file_type, storage_path, id)
      `);

    // Add filters
    if (subject) {
      query = query.eq("subjects.code", subject.toString().toUpperCase());
    }
    if (year) {
      query = query.eq("year", parseInt(year.toString()));
    }

    const { data, error } = await query;

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    // Transform response
    const grouped = (data || []).reduce((acc: any, paper: any) => {
      const key = `${paper.subjects?.code}-${paper.year}`;
      if (!acc[key]) {
        acc[key] = {
          subject: paper.subjects?.code,
          year: paper.year,
          level: paper.subjects?.level,
          sessions: {},
        };
      }

      if (!acc[key].sessions[paper.session]) {
        acc[key].sessions[paper.session] = {
          session: paper.session,
          papers: {},
        };
      }

      if (!acc[key].sessions[paper.session].papers[paper.paper]) {
        acc[key].sessions[paper.session].papers[paper.paper] = {
          paper: paper.paper,
          files: {},
        };
      }

      // Add file types
      (paper.paper_files || []).forEach((pf: any) => {
        acc[key].sessions[paper.session].papers[paper.paper].files[pf.file_type] = {
          file_type: pf.file_type,
          storage_path: pf.storage_path,
          id: pf.id,
        };
      });

      return acc;
    }, {});

    res.json({
      results: Object.values(grouped),
      count: Object.keys(grouped).length,
    });
  } catch (error) {
    console.error("Content search error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * GET /content/subjects
 * Get all available subjects
 */
router.get("/subjects", async (req: Request, res: Response) => {
  try {
    const { data, error } = await supabase
      .from("subjects")
      .select("id, code, level")
      .order("code");

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.json(data);
  } catch (error) {
    console.error("Subjects error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * GET /content/papers
 * Get papers by subject and/or year
 */
router.get("/papers", async (req: Request, res: Response) => {
  try {
    const { subject, year } = req.query;

    let query = supabase
      .from("papers")
      .select("year, session, paper, subjects!inner(code)");

    if (subject) {
      query = query.eq("subjects.code", subject.toString().toUpperCase());
    }
    if (year) {
      query = query.eq("year", parseInt(year.toString()));
    }

    const { data, error } = await query;

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    // Group by year and session
    const grouped = (data || []).reduce(
      (acc: any, item: any) => {
        const yearKey = `${item.year}`;
        if (!acc[yearKey]) {
          acc[yearKey] = {
            year: item.year,
            sessions: new Set<string>(),
            papers: new Set<string>(),
          };
        }
        acc[yearKey].sessions.add(item.session);
        acc[yearKey].papers.add(item.paper);
        return acc;
      },
      {} as Record<
        string,
        { year: number; sessions: Set<string>; papers: Set<string> }
      >
    );

    const result = Object.values(grouped).map((item: any) => ({
      year: item.year,
      sessions: Array.from(item.sessions),
      papers: Array.from(item.papers),
    }));

    res.json(result);
  } catch (error) {
    console.error("Papers error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

export default router;
