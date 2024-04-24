import { NextRequest, NextResponse } from 'next/server';

import { ChatOpenAI } from 'langchain/chat_models/openai';
import { BytesOutputParser } from 'langchain/schema/output_parser';
import { PromptTemplate } from 'langchain/prompts';
import { LLMChain } from 'langchain/chains';
import { apiLogger, categorizePrompt } from '@/utils/actions';
import { Message } from 'ai';
import { BlockRuleSchema } from '@/types/schemas';
import { createServerComponentClient } from '@/utils/supabaseServer';
import { evaluateMessageAgainstRules } from '@/utils/ruleHelpers';
import { personaAttributeLevel } from '@/types/index';
import { createPersonaPrompt} from '@/utils/personaAttributes';
import { supabaseAdmin } from '@/utils/supabaseAdminClient';

/**
 * Basic memory formatter that stringifies and passes
 * message history directly into the model.
 */
const formatMessage = (message: Message) => {
  return `${message.role}: ${message.content}`;
};

export async function POST(req: NextRequest) {
  const incomingData = await req.json();
  const supabase = createServerComponentClient();
  const { messages: rawMessages, company_id, user_id, chat_block_id } = incomingData;
  const messages: Message[] = rawMessages.map((message: any) => {
    return {
      role: message.role,
      content: message.content,
    };
  });

  //TODO send the persona_id so it grabs the right one not just the first one
  const { data: personaData, error: personaError } = await supabaseAdmin
    .from('chat_block_persona')
    .select('persona(*)')
    .eq('chat_block_id', chat_block_id)
    .limit(1)
    .single();

  if (personaError || !personaData) {
    console.error(personaError);
  }

  console.log(' personaData ', personaData );

  const attributesInput: personaAttributeLevel[] = Object.entries(
    personaData?.persona?.attributes!
  ).map(([key, value]) => {
    return {
      [key]: value as 0 | 1 | 2,
    };
  });

  // const generatedPrompt = generateSystemPrompt(attributesInput);
  const generatedPrompt = createPersonaPrompt(attributesInput);

  const TEMPLATE = `${generatedPrompt}
 
  Current conversation:
  {chat_history}
  
  User: {input}
  AI:`;

  const formattedPreviousMessages = rawMessages.slice(0, -1).map(formatMessage);
  const currentMessageContent = rawMessages[rawMessages.length - 1].content;

  const prompt = PromptTemplate.fromTemplate(TEMPLATE);
  const model = new ChatOpenAI({
    temperature: 0.8,
    openAIApiKey: process.env.OPENAI_API_KEY || '',
  });

  const outputParser = new BytesOutputParser();

  const categorization = await categorizePrompt(messages[messages.length - 1].content);

  apiLogger({
    data: {
      message: messages[messages.length - 1].content,
      category: categorization.category,
      complexity: categorization.complexity,
      topics: categorization.topics,
    },
    severity: 'INFO',
    type: 'CHAT_MESSAGE_CREATED',
    company_id: company_id,
    user_id: user_id,
  });

  const { data: rulesData, error: rulesDataError } = await supabase
    .from('block_rule')
    .select('*')
    .eq('company_id', company_id);

  if (rulesDataError) {
    console.error(rulesDataError);
    return null;
  }

  const { data: messageInsert, error: messageInsertError } = await supabaseAdmin
    .from('chat_message')
    .insert([
      {
        chat_block_id: chat_block_id,
        content: messages[messages.length - 1].content,
        user_id: user_id,
        type: 'USER',
        content_type: 'TEXT',
        category: categorization.category,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ])
    .select();

  if (messageInsertError) {
    console.error(messageInsertError);
    return null;
  } else {
    console.log('messageInsert', messageInsert);
  }

  const parsedRules = rulesData?.map(rule => BlockRuleSchema.parse(rule));

  const triggeredRules = await evaluateMessageAgainstRules(
    {
      message: messages[messages.length - 1],
      company_id: company_id,
      user_id: user_id,
      message_id: messageInsert[0].id,
    },
    parsedRules
  );

  if (triggeredRules.length > 0) {
    apiLogger({
      data: {
        message: messages[messages.length - 1].content,
        triggeredRules: triggeredRules,
      },
      severity: 'INFO',
      type: 'CHAT_MESSAGE_BLOCK_RULE_TRIGGERED',
      company_id: company_id,
      user_id: user_id,
    });
  }

  // Replace last message with substituted values from triggeredRules
  const newMessages = messages.map(message => {
    const newMessage = { ...message };

    triggeredRules.forEach(rule => {
      newMessage.content = newMessage.content.replace(rule.original, rule.substitution);
    });

    return {
      role: newMessage.role,
      content: newMessage.content,
    };
  });

  const chain = prompt.pipe(model).pipe(outputParser);

  const chainB = new LLMChain({
    prompt: prompt,
    llm: model,
  });

  const resB = await chainB.call({
    chat_history: formattedPreviousMessages.join('\n'),
    input: currentMessageContent,
  });

  if (resB) {
    const { data, error } = await supabase
      .from('chat_message')
      .insert([
        {
          chat_block_id: chat_block_id,
          content: resB.text,
          type: 'BOT',
          content_type: 'TEXT',
          category: categorization.category,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ])
      .select();

    if (error) {
      console.error(error);
      return;
    }
  }

  return NextResponse.json(resB);
}


