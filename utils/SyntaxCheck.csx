#r "nuget: Microsoft.CodeAnalysis.CSharp, 4.9.0"
#r "nuget: Microsoft.CodeAnalysis.Common, 4.9.0"
#r "nuget: Newtonsoft.Json, 13.0.3"

using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Newtonsoft.Json;

if (Args.Count < 1)
{
    Console.WriteLine("Usage: dotnet script SyntaxCheck.csx <file1.cs> <file2.cs> ...");
    return;
}

var results = new List<object>();

foreach (var filePath in Args)
{
    if (!File.Exists(filePath))
    {
        results.Add(new {
            type = "file-error",
            module = Path.GetFileName(filePath),
            obj = "",
            line = 0,
            column = 0,
            path = filePath,
            symbol = "file-not-found",
            message = $"File not found: {filePath}",
            message_id = ""
        });
        continue;
    }

    var code = File.ReadAllText(filePath);
    var tree = CSharpSyntaxTree.ParseText(code, path: filePath); // 👈 important: keeps path in diagnostics
    var diagnostics = tree.GetDiagnostics();

    foreach (var diag in diagnostics)
    {
        if (diag.Severity == DiagnosticSeverity.Error)
        {
            var lineSpan = diag.Location.GetLineSpan();
            results.Add(new {
                type = "syntax-error",
                module = Path.GetFileName(filePath),
                obj = "",
                line = lineSpan.StartLinePosition.Line + 1,
                column = lineSpan.StartLinePosition.Character + 1,
                path = filePath,
                symbol = "syntax-error",
                message = diag.GetMessage(),
                message_id = diag.Id
            });
        }
    }
}

Console.WriteLine(JsonConvert.SerializeObject(results, Formatting.Indented));
