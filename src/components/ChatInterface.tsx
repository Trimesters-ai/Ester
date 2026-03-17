import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, ChevronDown } from 'lucide-react';
import { useStore } from '../store/useStore';
import { getTimeAgo } from '../utils/timeAgo';
import ReactMarkdown from 'react-markdown';
import { streamOpenAIChat } from '../utils/openai';
import * as chrono from 'chrono-node';
import { isValid } from 'date-fns';
import { formatInTimeZone } from 'date-fns-tz';

const GENERIC_STARTERS = [
  "How does my sleep quality affect my postpartum recovery?",
  "What does my recovery score mean for breastfeeding?",
  "Should I start exercising based on my current strain?",
  "Is my heart rate variability normal for postpartum?",
  "How can I improve my sleep while caring for my baby?"
];

const DELIVERY_DATE_PROMPT = "My baby was delivered on YYYY-MM-DD";
const SLEEP_PROMPT = "How can I improve my sleep while caring for my baby?";
const ACTIVITY_PROMPT = "When is it safe to start exercising postpartum?";
const HRV_PROMPT = "Is my heart rate variability normal for postpartum?";
const BREASTFEEDING_PROMPT = "What does my recovery score mean for breastfeeding?";

export const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const { messages, addMessage, user } = useStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollDown, setShowScrollDown] = useState(false);
  
  const [suggestions, setSuggestions] = useState<string[]>([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Show scroll-to-bottom button if not at bottom
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      console.log('[Scroll Debug] scrollTop:', scrollTop, 'clientHeight:', clientHeight, 'scrollHeight:', scrollHeight);
      setShowScrollDown(scrollTop + clientHeight < scrollHeight - 30); // 30px threshold
    };
    container.addEventListener('scroll', handleScroll);
    // Check on mount and when messages change
    handleScroll();
    return () => container.removeEventListener('scroll', handleScroll);
  }, [messages.length]);

  useEffect(() => {
    scrollToBottom();
    inputRef.current?.focus();
  }, [messages]);

  // Helper to update suggestions based on latest context
  const updateSuggestions = (latestContent?: string) => {
    if (!user?.postpartumDate) {
      setSuggestions([
        DELIVERY_DATE_PROMPT,
        SLEEP_PROMPT,
        ACTIVITY_PROMPT
      ]);
    } else if (messages.length === 0) {
      setSuggestions(GENERIC_STARTERS.slice(0, 3));
    } else {
      const lastMessage = (latestContent ?? messages[messages.length - 1]?.content ?? '').toLowerCase();
      const pool = [SLEEP_PROMPT, ACTIVITY_PROMPT, HRV_PROMPT, BREASTFEEDING_PROMPT];
      if (lastMessage.includes("sleep")) {
        setSuggestions([ACTIVITY_PROMPT, HRV_PROMPT]);
      } else if (lastMessage.includes("exercise") || lastMessage.includes("activity")) {
        setSuggestions([SLEEP_PROMPT, HRV_PROMPT]);
      } else if (lastMessage.includes("hrv") || lastMessage.includes("heart rate variability")) {
        setSuggestions([SLEEP_PROMPT, ACTIVITY_PROMPT]);
      } else {
        setSuggestions(pool.slice(0, 3));
      }
    }
  };

  // Update suggestions on any message change (not just length)
  useEffect(() => {
    updateSuggestions();
  }, [user?.postpartumDate, messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !user) return;

    // Clear any previous error when user tries again
    setErrorMessage(null);

    const userMessage = {
      id: crypto.randomUUID(),
      content: input,
      timestamp: new Date().toISOString(),
      userId: user.id,
      isAI: false,
    };

    addMessage(userMessage);
    setInput('');
    setIsLoading(true);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 10);

    const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    console.log('Detected user timezone:', userTimezone);

    try {
      const namePhraseRegex = /(?:my name is|i am|i'm|this is)\s+([A-Z][a-zA-Z'-]{1,30})\b(?!\s+(to|for|and|with|on|in|at|by|from|of|about|as|into|like|through|after|over|between|out|against|during|without|before|under|around|among))/gi;
      let nameMatch: RegExpExecArray | null = null;
      let lastName: string | null = null;
      while ((nameMatch = namePhraseRegex.exec(userMessage.content)) !== null) {
        lastName = nameMatch[1];
      }
      if (lastName) {
        const cleanedName = lastName.trim();
        const commonWords = [
          'Ready', 'Start', 'Send', 'Hello', 'Cancel', 'Save', 'Test', 'Chat', 'User', 'Anonymous', 'Thank', 'You',
          'Today', 'Tomorrow', 'Yesterday', 'Morning', 'Evening', 'Night', 'Afternoon', 'Week', 'Month', 'Year', 'Day',
          'To', 'For', 'And', 'With', 'On', 'In', 'At', 'By', 'From', 'Of', 'About', 'As', 'Into', 'Like', 'Through', 'After', 'Over', 'Between', 'Out', 'Against', 'During', 'Without', 'Before', 'Under', 'Around', 'Among',
          'Yes', 'No', 'Okay', 'Ok', 'Great', 'Sure', 'Fine', 'Good', 'Bad', 'Awesome', 'Cool', 'Ready', 'Go', 'Stop', 'Continue', 'Begin', 'End', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'
        ];
        if (!commonWords.map(w => w.toLowerCase()).includes(cleanedName.toLowerCase()) && cleanedName.length > 1) {
          if (user.name !== cleanedName) {
            console.log('Updating user name from', user.name, 'to', cleanedName);
            useStore.getState().setUser({ ...user, name: cleanedName });
          } else {
            console.log('User name unchanged:', cleanedName);
          }
        } else {
          console.log('Rejected extracted name as invalid:', cleanedName);
        }
      }

      let detectedDate: Date | null = null;
      const results = chrono.parse(userMessage.content, new Date(), { forwardDate: true });
      if (results.length > 0 && results[0].start.isCertain('day') && results[0].start.isCertain('month')) {
        let chronoDate = results[0].start.date();
        if (!results[0].start.isCertain('year')) {
          chronoDate.setFullYear(new Date().getFullYear());
        }
        detectedDate = chronoDate;
        if (isValid(detectedDate)) {
          const tz = userTimezone;
          const localDate = new Date(detectedDate.getFullYear(), detectedDate.getMonth(), detectedDate.getDate());
          const isoWithTz = formatInTimeZone(localDate, tz, "yyyy-MM-dd'T'HH:mm:ssXXX");
          if (user.postpartumDate !== isoWithTz) {
            console.log('Updating postpartum date from', user.postpartumDate, 'to', isoWithTz);
            useStore.getState().setUser({ ...user, postpartumDate: isoWithTz });
          } else {
            console.log('Postpartum date unchanged:', isoWithTz);
          }
        } else {
          console.log('Rejected postpartum date from chrono-node as invalid:', detectedDate);
        }
      } else {
        const isoDatePattern = /^\d{4}-\d{2}-\d{2}(?:[Tt ][0-9:.\-+Z]*)?$/;
        if (isoDatePattern.test(userMessage.content.trim())) {
          const parsedDate = new Date(userMessage.content.trim());
          if (!isNaN(parsedDate.getTime()) && isValid(parsedDate)) {
            const tz = userTimezone;
            const localDate = new Date(parsedDate.getFullYear(), parsedDate.getMonth(), parsedDate.getDate());
            const isoWithTz = formatInTimeZone(localDate, tz, "yyyy-MM-dd'T'HH:mm:ssXXX");
            if (user.postpartumDate !== isoWithTz) {
              console.log('Updating postpartum date from', user.postpartumDate, 'to', isoWithTz);
              useStore.getState().setUser({ ...user, postpartumDate: isoWithTz });
            } else {
              console.log('Postpartum date unchanged:', isoWithTz);
            }
          } else {
            console.log('Rejected postpartum date as invalid:', userMessage.content.trim());
          }
        }
      }

      const aiMessageId = crypto.randomUUID();
      addMessage({
        id: aiMessageId,
        content: '',
        timestamp: new Date().toISOString(),
        userId: user.id,
        isAI: true,
      });

      const prevMessages = [...messages, userMessage];
      let aiContent = '';
      let systemPromptPrefix = '';
      if (user?.name) {
        systemPromptPrefix += `The user's name is ${user.name}. Please address them by their name in your responses. `;
      }
      if (user?.postpartumDate) {
        systemPromptPrefix += `The user's baby was delivered on ${user.postpartumDate}. Use this information to provide personalized postpartum advice. The user's local timezone is ${userTimezone}. If the user provides a new date in YYYY-MM-DD or W3C/ISO 8601 format, acknowledge it and use it for future context. `;
      } else {
        systemPromptPrefix += `The user's baby's delivery date has not been provided. If it is contextually appropriate, gently and empathetically ask the user when their baby was delivered, and request the date in YYYY-MM-DD (W3C/ISO 8601) format. Use bedside manner and sensitivity. The user's local timezone is ${userTimezone}. Do not ask repeatedly. When the user provides a date in this format, acknowledge it and use it for future context. `;
      }
      const stream = await streamOpenAIChat({ messages: prevMessages, user, systemPromptPrefix });
      for await (const chunk of stream) {
        aiContent += chunk;
        useStore.getState().setMessages([
          ...prevMessages,
          { id: aiMessageId, content: aiContent, timestamp: new Date().toISOString(), userId: user.id, isAI: true }
        ]);
      }
      updateSuggestions(aiContent);
      setIsLoading(false);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    } catch (error: any) {
      console.error('Error sending message:', error);
      const message =
        typeof error?.message === 'string'
          ? error.message
          : 'Something went wrong while talking to the AI. Please try again.';
      // Special-case missing API key to give a clear instruction
      if (message.includes('Missing OpenAI API key')) {
        setErrorMessage('Missing OpenAI API key. Please add your key in the Profile tab or configure VITE_OPENAI_API_KEY in your environment.');
      } else if (message.toLowerCase().includes('openai api error')) {
        setErrorMessage('The OpenAI API returned an error. Please check that your API key is valid and has available quota.');
      } else {
        setErrorMessage(message);
      }
      setIsLoading(false);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };

  const handleStarterClick = (starter: string) => {
    setInput(starter);

  };

  return (
    <div className="flex flex-col w-full min-h-screen pt-20 pb-32 bg-white relative">
      {/* Scroll-to-bottom icon (fixed to viewport, but only when thread is scrollable) */}
      {/* Scroll-to-bottom icon: semi-transparent, only when needed */}
      {showScrollDown && messages.length > 0 && (
        <button
          className="fixed bottom-28 right-6 z-30 bg-white/60 rounded-full p-3 shadow-lg border border-green-200 transition-all duration-200 backdrop-blur-sm opacity-50 hover:opacity-90"
          style={{ boxShadow: '0 4px 16px rgba(80,0,120,0.10)' }}
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
        >
          <ChevronDown className="w-7 h-7 text-[#2d8059]" />
        </button>
      )}
      {/* Message thread area: fixed height, scrollable */}
      <div
        className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px]"
        style={{ paddingBottom: '90px', height: 'calc(100vh - 11rem)' }}
        ref={scrollContainerRef}
      >

        {errorMessage && (
          <div className="mb-2 max-w-5xl mx-auto">
            <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
              {errorMessage}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.isAI ? 'justify-start' : 'justify-end'
            } animate-fade-in`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 shadow-sm transform transition-all duration-200 hover:scale-[1.01] ${
                message.isAI
                  ? 'bg-[#e8f5e9] text-[#2d8059]'
                  : 'bg-blue-100 text-blue-900'
              }`}
            >
              {message.isAI ? (
                <ReactMarkdown
                  className="prose prose-sm max-w-none text-[#2d8059]"
                  components={{
                    p: ({ children }) => <p style={{ marginBottom: '1.25em' }}>{children}</p>,
                    br: () => <br style={{ marginBottom: '0.5em' }} />,
                  }}
                >
                  {
                    (function preprocessMarkdown(text: string): string {
                      // Step 1: Replace ISO dates with user-friendly dates
                      let content = text.replace(/\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?)?)\b/g, (dateStr) => {
                        const d = new Date(dateStr);
                        if (!isNaN(d.getTime())) {
                          try {
                            const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
                            return formatInTimeZone(d, userTimezone, 'MMMM d, yyyy');
                          } catch {
                            return d.toLocaleDateString(undefined, {
                              year: 'numeric',
                              month: 'long',
                              day: 'numeric',
                            });
                          }
                        }
                        return dateStr;
                      });
                      // Step 2: Add extra newline after each bullet point
                      content = content.replace(/^(\s*[-*] .+)$/gm, '$1\n');
                      // Step 3: Add double newline after sentence-ending punctuation if followed by a capital/question word (for paragraph breaks)
                      content = content.replace(/([.!?])(\s+)(?=(How|What|Why|When|Where|Who|Which|Is|Are|Do|Does|Did|Can|Could|Would|Will|Shall|Should|May|Might|Must|[A-Z]))/g, '$1\n\n');
                      return content;
                    })(message.content)
                  }
                </ReactMarkdown>
              ) : (
                <p className="text-sm leading-relaxed">{message.content}</p>
              )}
              <p className="text-xs mt-1 opacity-70">
                {getTimeAgo(message.timestamp)}
              </p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Render dynamic suggestions above the input box */}
      {/* Centered suggestions for initial load */}
      {messages.length === 0 && suggestions.length > 0 ? (
        <div className="fixed inset-0 flex flex-col items-center justify-center z-10 pointer-events-none">
          <div className="bg-white/90 px-10 py-8 flex flex-col items-center max-w-lg w-full pointer-events-auto" data-component-name="ChatInterface">
            <h2 className="mb-4 text-xl font-semibold text-[#2d8059]" data-component-name="ChatInterface">Try getting started by asking...</h2>
            <div className="flex flex-wrap gap-3 justify-center">
              {suggestions.map((suggestion, idx) => {
                const words = suggestion.split(' ');
                const displayText = words.length > 4 ? words.slice(0, 4).join(' ') + '...' : suggestion;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleStarterClick(suggestion)}
                    className="p-3 px-6 bg-[#f1f8e9] hover:bg-[#e8f5e9] rounded-full text-base text-[#2d8059] border border-green-200 transition-colors duration-200 ease-in-out shadow-md font-medium"
                    style={{ marginBottom: 6 }}
                    title={suggestion}
                  >
                    {displayText}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : suggestions.length > 0 && (
        <div className="w-full max-w-5xl mx-auto mb-2 px-6">
          <div className="flex flex-wrap gap-2 justify-start">
            {suggestions.map((suggestion, idx) => {
              const words = suggestion.split(' ');
              const displayText = words.length > 4 ? words.slice(0, 4).join(' ') + '...' : suggestion;
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleStarterClick(suggestion)}
                  className="p-2 px-4 bg-[#f1f8e9] hover:bg-[#e8f5e9] rounded-full text-sm text-[#2d8059] border border-green-200 transition-colors duration-200 ease-in-out shadow-sm"
                  style={{ marginBottom: 4 }}
                  title={suggestion}
                >
                  {displayText}
                </button>
              );
            })}
          </div>
        </div>
      )}
      <form
        onSubmit={handleSubmit}
        className="fixed bottom-0 left-0 w-full flex justify-center z-20"
        style={{ boxShadow: '0 -4px 12px rgba(0,0,0,0.1)' }}
      >
        <div className="flex items-center space-x-3 w-full p-4 bg-white">
          <div className="flex items-center space-x-3 w-full max-w-5xl mx-auto bg-white px-6 py-4 rounded-xl shadow-sm">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 rounded-full px-5 py-3 border-2 border-green-200 focus:outline-none focus:ring-2 focus:ring-[#bcc547] focus:border-[#bcc547] transition-all duration-200 shadow-sm text-gray-800 placeholder-gray-500"
              disabled={isLoading}
              autoFocus
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-[#2d8059] text-white rounded-full p-3 hover:bg-[#1f5c40] disabled:opacity-50 transition-colors duration-200 hover:shadow-md"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
