import type { VercelRequest, VercelResponse } from '@vercel/node';

const OPENAI_API_URL = 'https://api.openai.com/v1/responses';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const apiKey = process.env.OPENAI_API_KEY || process.env.VITE_OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'Missing OPENAI_API_KEY on the server.' });
  }

  try {
    const {
      messages,
      whoopData,
      max_tokens = 512,
      temperature = 0.7,
      systemPromptPrefix = '',
    } = req.body as {
      messages: any[];
      whoopData?: any;
      max_tokens?: number;
      temperature?: number;
      systemPromptPrefix?: string;
    };

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Invalid or missing messages array.' });
    }

    const SYSTEM_PROMPT = `You are Ester, a compassionate, medically-informed postpartum recovery assistant. You support all individuals through their postpartum journey, including those who have experienced live births, stillbirths, or pregnancy losses. Never assume the outcome of the user's pregnancy. Only reference the baby if the user has explicitly mentioned their baby's status. Never offer congratulations unless the user has explicitly shared positive news about their baby. Be especially sensitive to the full spectrum of postpartum experiences and emotions. Provide the same level of care and support regardless of pregnancy outcome. Use the user's Whoop health data (if provided) to give personalized, supportive, and clear advice. Never ask the user to provide a date in a specific format such as YYYY-MM-DD, ISO, or W3C. Always ask for and reference dates in a natural, conversational way.`;

    const INSTRUCTIONS = `Format all responses in markdown for readability. Use Whoop health data context if available. Be empathetic, clear, and actionable. Be extremely mindful that users may have experienced different pregnancy outcomes, including pregnancy loss. Never assume the status of the user's baby. Only reference the baby if the user has mentioned them first. Never offer congratulations or assume a positive outcome unless explicitly stated by the user. Be particularly sensitive when discussing postpartum recovery, as the emotional and physical needs can vary greatly. Never ask the user to provide a date in a specific format such as YYYY-MM-DD, ISO, or W3C. Always ask for and reference dates in a natural, conversational way.

For new conversations, follow these steps:
1. If this is the first message in the conversation, introduce yourself warmly: "Hello, I'm Ester, your postpartum recovery assistant. I'm here to support you through your postpartum journey. May I know your name? (Only if you feel comfortable sharing)"
2. After learning their name, ask for their postpartum date in a natural way: "Thank you for sharing, [Name]. Could you tell me when your postpartum period began? This helps me provide you with the most relevant support."
3. If they don't provide a name, continue the conversation naturally without pressing for it.
4. If they don't provide a postpartum date, you can ask once more later in the conversation, but don't be insistent.
5. Remember to be sensitive and understanding if they choose not to share this information.`;

    // Build the single text input, mirroring the client utility
    let input = (systemPromptPrefix ? systemPromptPrefix + '\n' : '') + SYSTEM_PROMPT + '\n';
    if (whoopData) {
      input += `User's latest Whoop health data: ${JSON.stringify(whoopData)}\n`;
    }
    for (const m of messages) {
      input += (m.isAI ? 'Assistant: ' : 'User: ') + m.content + '\n';
    }

    const commitId =
      process.env.VERCEL_GIT_COMMIT_SHA || process.env.VITE_COMMIT_ID || 'vercel-deployment';

    const response = await fetch(OPENAI_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        input,
        instructions: INSTRUCTIONS,
        temperature,
        max_output_tokens: max_tokens,
        stream: false,
        metadata: {
          commit: commitId,
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(response.status).json({ error: 'OpenAI API error', details: errorText });
    }

    const data = await response.json();

    // Responses API: extract plain text content defensively
    let text = '';
    try {
      const output = (data.output || data.response?.output || [])[0];
      const content = output?.content || output?.contents || [];
      const first = content[0];
      text =
        first?.text ||
        first?.output_text?.text ||
        first?.output_text?.content?.[0]?.text ||
        JSON.stringify(data);
    } catch {
      text = JSON.stringify(data);
    }

    return res.status(200).json({ text });
  } catch (err: any) {
    console.error('Server chat error:', err);
    return res.status(500).json({
      error: 'Unexpected server error while talking to OpenAI',
      details: err?.message || String(err),
    });
  }
}

