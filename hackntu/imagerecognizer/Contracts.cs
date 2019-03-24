using System;

namespace VisualRecognition
{
    internal static class Contracts
    {
        public static void Check<TException>(bool valid, string message = "", params object[] args)
            where TException : Exception
        {
            if (!valid)
            {
                var e = (TException)Activator.CreateInstance(typeof(TException), string.Format(message, args));
                throw e;
            }
        }

        public static void Check(bool valid, string message = "", params object[] args)
        {
            Check<ArgumentException>(valid, message, args);
        }

        public static void CheckValue(string message, params object[] values)
        {
            foreach (var value in values)
                Check(value != null, message);
        }
    }
}
